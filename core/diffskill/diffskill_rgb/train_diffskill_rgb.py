from tqdm import tqdm
import torch
import json
import os
from chester import logger
from chester.utils import set_ipdb_debugger
from plb.envs.mp_wrapper import make_mp_envs

from core.utils.utils import set_random_seed
from core.diffskill.args import get_args


def prepare_buffer(args, device):
    from core.diffskill.imitation_buffer import ImitationReplayBuffer, filter_buffer_nan
    from core.diffskill.utils import load_target_info
    # Load buffer
    buffer = ImitationReplayBuffer(args)
    buffer.load(args.dataset_path)
    filter_buffer_nan(buffer)
    buffer.generate_train_eval_split(filter=args.filter_buffer)
    target_info = load_target_info(args, device, load_set=False)
    buffer.__dict__.update(**target_info)
    return buffer


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    if 'debug' in exp_name:
        set_ipdb_debugger()
        import faulthandler
        faulthandler.enable()
    args = get_args(cmd=False)
    args.__dict__.update(**arg_vv)
    set_random_seed(args.seed)

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)

    # Need to make the environment before moving tensor to torch
    env = make_mp_envs(args.env_name, args.num_env, args.seed)
    print('------------Env created!!--------')

    args.cached_state_path = env.getattr('cfg.cached_state_path', 0)
    action_dim = env.getattr('taichi_env.primitives.action_dim')[0]
    args.action_dim = action_dim

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    from core.diffskill.agent import Agent
    from core.diffskill.utils import aggregate_traj_info, dict_add_prefix
    from core.diffskill.eval_helper import eval_skills, eval_plan

    device = 'cuda'
    buffer = prepare_buffer(args, device)

    # # ----------preparation done------------------
    agent = Agent(args, None, num_tools=args.num_tools, device=device)
    print('Agent created')
    if args.resume_path is not None:
        agent.load(args.resume_path, args.load_modules)
    elif args.vae_resume_path is not None:
        agent.load(args.vae_resume_path, ['vae'])

    if 'vae' not in args.train_modules:
        agent.vae.generate_cached_buffer(buffer)

    for epoch in range(args.il_num_epoch):
        set_random_seed(
            (args.seed + 1) * (epoch + 1))  # Random generator may change since environment is not deterministic and may change during evaluation
        infos = {'train': [], 'eval': []}
        for mode in ['train', 'eval']:
            epoch_tool_idxes = [buffer.get_epoch_tool_idx(epoch, tid, mode) for tid in range(args.num_tools)]
            for batch_tools_idx in tqdm(zip(*epoch_tool_idxes), desc=mode):
                data_batch = buffer.sample_tool_transitions(batch_tools_idx, epoch, device)
                if mode == 'eval':
                    with torch.no_grad():
                        train_info = agent.train(data_batch, mode=mode, epoch=epoch)
                else:
                    train_info = agent.train(data_batch, mode=mode, epoch=epoch)

                infos[mode].append(train_info)
            infos[mode] = aggregate_traj_info(infos[mode], prefix=None)
            infos[mode] = dict_add_prefix(infos[mode], mode + '/')

        agent.update_best_model(epoch, infos['eval'])

        if epoch % args.il_eval_freq == 0:
            skill_info, vae_info, plan_info = {}, {}, {}

            # Evaluate skills
            if args.eval_skill and 'policy' in args.train_modules:
                skill_traj, skill_info = eval_skills(args, env, agent, epoch)
            if 'vae' in args.train_modules:
                from core.diffskill.eval_helper import eval_vae
                vae_info = eval_vae(args, buffer, agent, epoch=epoch, all_obses=buffer.buffer['obses'][buffer.eval_idx])

            # Plan
            if args.eval_plan and epoch % args.il_eval_freq == 0:
                agent.load_best_model()  # Plan with the best model
                plan_info = eval_plan(args, env, agent, epoch, demo=False)
                if 'best_trajs' in plan_info:
                    del plan_info['best_trajs']
                agent.load_training_model()  # Return training

            # Logging
            logger.record_tabular('epoch', epoch)
            all_info = {}

            plan_info = dict_add_prefix(plan_info, 'plan/')

            # all_info.update(**train_info)
            all_info.update(**skill_info)
            all_info.update(**vae_info)
            all_info.update(**plan_info)
            [all_info.update(**infos[mode]) for mode in infos.keys()]
            for key, val in all_info.items():
                logger.record_tabular(key, val)
            logger.dump_tabular()

            # Save model
            if epoch % args.il_eval_freq == 0:
                agent.save(os.path.join(logger.get_dir(), f'agent_{epoch}.ckpt'))
    env.close()


if __name__ == '__main__':
    env = make_mp_envs('CutRearrange-v1', 1, 0)
    print('Env created')
