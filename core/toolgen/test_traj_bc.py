import pickle
from tqdm import tqdm
import torch
import json
import os
from chester import logger
from chester.utils import set_ipdb_debugger
from core.utils.utils import set_random_seed
from core.toolgen.args import get_args

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
    from core.toolgen.train_model import prepare_buffer
    if args.action_replay or args.use_gt_tool_and_reset or args.use_contact:
        buffer = prepare_buffer(args, device)
    else:
        buffer = None
    # # ----------preparation done------------------
    agent = BCAgent(args, device=device)
    infos = {}
    print('Agent created')
    if args.resume_path is not None:
        agent.load(args.resume_path)
    epoch = 0
    rollout_info = {}

    # Logging
    logger.record_tabular('epoch', epoch)
    all_info = {}
    if True:
        rollout_traj, rollout_info = eval_traj(args, env, agent, epoch, buffer=buffer)
        with open(os.path.join(logger.get_dir(), f'traj_{epoch}.pkl'), 'wb') as f:
            pickle.dump(rollout_traj, f)
        rollout_info = dict_add_prefix(rollout_info, '/')
        all_info.update(**rollout_info)
    [all_info.update(**infos[mode]) for mode in infos.keys()]
    for key, val in all_info.items():
        logger.record_tabular(key, val)
    if args.use_wandb:
        wandb.log(all_info, step=epoch)
    logger.dump_tabular()

    # Save model
    agent.save(os.path.join(logger.get_dir(), f'agent_{epoch}.ckpt'))
    agent.save_train_stats(os.path.join(logger.get_dir(), f'train_buffer_stats_{epoch}.pkl'))
    env.close()
