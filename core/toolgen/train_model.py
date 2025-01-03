from tqdm import tqdm
import torch
import json
import os
from chester import logger
from chester.utils import set_ipdb_debugger
from core.diffskill.utils import get_obs_goal_match_buffer
from core.utils.utils import set_random_seed
from core.toolgen.args import get_args


def prepare_buffer(args, device):
    from core.toolgen.vat_buffer import VatMartDataset, filter_buffer_nan
    from core.diffskill.utils import load_target_info
    # Load buffer
    buffer = VatMartDataset(args)
    buffer.load(args.dataset_path)
    filter_buffer_nan(buffer)
    buffer.generate_train_eval_split(filter=args.filter_buffer)
    target_info = load_target_info(args, device, load_set=False)
    buffer.__dict__.update(**target_info)
    if 'goal_as_flow' in args.actor_type:
        matched_goals = get_obs_goal_match_buffer(buffer)
        buffer.__dict__.update(matched_goals=matched_goals)
    buffer.compute_stats()
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
    # if args.profiling:
    # from plb.envs import make
    # env = make(args.env_name)
    # args.cached_state_path = env.cfg.cached_state_path
    # action_dim = env.taichi_env.primitives.action_dim
    # args.action_dim = action_dim
    # else:
    from plb.envs.mp_wrapper import make_mp_envs
    env = make_mp_envs(args.env_name, args.num_env, args.seed, loss_type=args.env_loss_type)
    print('------------Env created!!--------')

    args.cached_state_path = env.getattr('cfg.cached_state_path', 0)
    args.pcl_dir_path = env.getattr('cfg.pcl_dir_path', 0)
    action_dim = env.getattr('taichi_env.primitives.action_dim')[0]
    args.action_dim = action_dim

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    if args.use_wandb:
        import wandb
        wandb.init(project='ToolRep',
                   group=args.wandb_group,
                   name=exp_name,
                   resume='allow', id=None,
                   settings=wandb.Settings(start_method='thread'))
        wandb.config.update(args, allow_val_change=True)

    from core.toolgen.bc_agent import BCAgent
    from core.diffskill.utils import aggregate_traj_info, dict_add_prefix
    from core.toolgen.eval_helper import eval_traj

    device = 'cuda'
    buffer = prepare_buffer(args, device)

    # # ----------preparation done------------------
    if args.train_model == 'bc':
        agent = BCAgent(args, device=device)
    else:
        raise NotImplementedError
    print('Agent created')
    if args.resume_path is not None:
        agent.load(args.resume_path, args.load_modules)


    for epoch in range(args.il_num_epoch):
        set_random_seed(
            (args.seed + 1) * (epoch + 1))  # Random generator may change since environment is not deterministic and may change during evaluation
        infos = {'train': [], 'eval': []}
        for mode in ['train', 'eval']:
            if args.actor_type == 'tfn_twostep' or args.actor_type == 'tfn_backflow' or args.actor_type == 'tfn_goal_as_flow':
                epoch_tool_idxes = [buffer.get_epoch_tool_idx(epoch, tid, mode) for tid in args.train_tool_idxes]
                for batch_tools_idx in tqdm(zip(*epoch_tool_idxes), desc=mode):
                    data_batch = buffer.sample_transition_twostep(batch_tools_idx, device)
                    train_info, train_plot_info = agent.train(data_batch, mode=mode)
                    infos[mode].append(train_info)
            else:
                epoch_tool_traj_idxes = [buffer.get_epoch_tool_traj_idx(epoch, tid, mode) for tid in args.train_tool_idxes]# range(args.num_tools)]
                print(epoch_tool_traj_idxes)
                for batch_tools_idx in tqdm(zip(*epoch_tool_traj_idxes), desc=mode):
                    data_batch = buffer.sample_transition_openloop(batch_tools_idx, device, use_contact=args.use_contact)
                    if mode == 'eval':
                        with torch.no_grad():
                            train_info, train_plot_info = agent.train(data_batch, mode=mode, epoch=epoch, ret_plot=epoch % args.il_eval_freq == 0)
                        # chamfer_info = agent.get_tool_reset_errors(data_batch)
                        # train_info.update(**chamfer_info)
                    else:
                        train_info, train_plot_info = agent.train(data_batch, mode=mode, epoch=epoch, ret_plot=epoch % args.il_eval_freq == 0)
                        # if args.train_model == 'bc':
                            # agent.update_train_stats(buffer.stats)
                    infos[mode].append(train_info)
            infos[mode] = aggregate_traj_info(infos[mode], prefix=None)
            infos[mode] = dict_add_prefix(infos[mode], mode + '/')
            # Wandb logging after each epoch
            if args.use_wandb:
                wandb.log(infos[mode], step=epoch)
                # Only log plots once in a whle
                if epoch % args.il_eval_freq == 0:
                    for key in train_plot_info.keys():
                        for i, plot in enumerate(train_plot_info[key]):
                            print('epoch', epoch, 'plot', i)
                            wandb.log({f'{mode}/{key}_{i}': wandb.Html(plot)}, step=epoch)

        agent.update_best_model(epoch, infos['eval'])
        if args.use_wandb:
            wandb.log(agent.best_dict, step=epoch)

        if epoch % args.il_eval_freq == 0:
            rollout_info = {}
            agent.load_best_model()  # Plan with the best model
            agent.load_training_model()  # Return training

            # Logging
            logger.record_tabular('epoch', epoch)
            all_info = {}
            with torch.no_grad():
                rollout_traj, rollout_info = eval_traj(args, env, agent, epoch, buffer=buffer)
                rollout_info = dict_add_prefix(rollout_info, '/')
                all_info.update(**rollout_info)
            [all_info.update(**infos[mode]) for mode in infos.keys()]
            for key, val in all_info.items():
                logger.record_tabular(key, val)
            if args.use_wandb:
                wandb.log(all_info, step=epoch)
            logger.dump_tabular()

            # Save model
            if epoch % args.il_eval_freq == 0:
                agent.save(os.path.join(logger.get_dir(), f'agent_{epoch}.ckpt'))
                # if args.train_model == 'bc':
                #     agent.save_train_stats(os.path.join(logger.get_dir(), f'train_buffer_stats_{epoch}.pkl'))
    env.close()