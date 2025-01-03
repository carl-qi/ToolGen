from core.diffskill.sampler import sample_traj, sample_traj_solver
from core.diffskill.imitation_buffer import ImitationReplayBuffer
from core.diffskill.args import get_args
from core.diffskill.utils import calculate_performance, calculate_performance_buffer, visualize_dataset
from core.diffskill.utils import visualize_trajs
from core.diffskill.env_spec import get_num_traj, set_render_mode
from core.utils.utils import set_random_seed
from plb.envs import make
import pickle
import random
import numpy as np
import torch
import json
import os
from chester import logger

def eval_training_traj(traj_ids, args, buffer, env, agent, target_imgs, np_target_imgs, np_target_mass_grids, save_name):
    horizon = 50
    trajs = []
    demo_obses = []
    demo_target_ious = []
    for traj_id in traj_ids:
        init_v = int(buffer.buffer['init_v'][traj_id * horizon])
        target_v = int(buffer.buffer['target_v'][traj_id * horizon])
        reset_key = {'init_v': init_v, 'target_v': target_v}
        tid = buffer.get_tid(buffer.buffer['action_mask'][traj_id * horizon])
        traj = sample_traj(env, agent, reset_key, tid)
        traj['target_img'] = np_target_imgs[reset_key['target_v']]
        demo_obs = buffer.buffer['obses'][traj_id * horizon: traj_id * horizon + horizon]
        demo_obses.append(demo_obs)
        demo_target_ious.append(buffer.buffer['target_ious'][traj_id * horizon + horizon - 1])
        print(f'tid: {tid}, traj_id: {traj_id}, reward: {np.sum(traj["rewards"])}')
        trajs.append(traj)
    demo_obses = np.array(demo_obses)

    agent_ious = np.array([traj['target_ious'][-1, 0] for traj in trajs])
    demo_target_ious = np.array(demo_target_ious)
    logger.log('Agent ious: {}, Demo ious: {}'.format(np.mean(agent_ious), np.mean(demo_target_ious)))

    visualize_trajs(trajs, 4, key='info_normalized_performance', save_name=os.path.join(logger.get_dir(), save_name),
                    overlay_target=False, demo_obses=demo_obses[:, :, :, :, :3])
    info = {'agent_iou': np.mean(agent_ious), 'demo_iou': np.mean(demo_target_ious)}
    return trajs, info

def check_correct_tid(env_name, init_v, target_v, tid):
    if env_name =='CutRearrange-v1' or env_name == 'CutRearrange-v2':
        correct_tid = 0 if init_v % 3 == 0 else 1
        return tid == correct_tid
    elif env_name =='CutRearrangeSpread-v1':
        if init_v < 200:
            correct_tid = 0
        elif init_v < 400:
            correct_tid = 1
        elif init_v < 600 or init_v >= 620:
            correct_tid = 2
        else:
            return True
        return tid == correct_tid
    elif 'Writer' in env_name:
        if init_v < get_num_traj(env_name) // 2:
            correct_tid = 0
        else:
            correct_tid = 1
        return tid == correct_tid
    elif env_name =='MultiTool-v1':
        if init_v < 200:
            correct_tid = 2
        elif init_v < 400:
            correct_tid = 3
        elif init_v < 600:
            correct_tid = 2
        elif init_v < 800:
            correct_tid = 4
        elif init_v < 1000:
            correct_tid = 0
        elif init_v < 1200:
            correct_tid = 1
        else:
            return True
        return tid == correct_tid
    else:
        return True
def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    from plb.engine.taichi_env import TaichiEnv
    from plb.optimizer.solver import Solver
    args = get_args(cmd=False)
    args.__dict__.update(**arg_vv)

    set_random_seed(args.seed)

    device = 'cuda'

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)
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
    # ----------preparation done------------------
    buffer = ImitationReplayBuffer(args)
    obs_channel = len(args.img_mode) * args.frame_stack

    env = make(args.env_name, nn=(args.algo == 'nn'), loss_type=args.env_loss_type, tool_model_path=args.tool_model_path)
    env.seed(args.seed)
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    set_render_mode(env, args.env_name, 'mesh')

    if args.data_name == 'loading':
        with open(args.loading_traj_path, 'rb') as f:
            trajs = pickle.load(f)
        init_vs = [traj['init_v'] for traj in trajs]
        target_vs = [traj['target_v'] for traj in trajs]
        tids = [traj['selected_tools'].item() for traj in trajs]
        reset_states = [traj['waypoints'][0] for traj in trajs]
        solver = Solver(args, taichi_env, (0,), return_dist=True)
        args.dataset_path = os.path.join(logger.get_dir(), 'dataset.gz')
        from core.diffskill.env_spec import get_tool_spec
        tool_spec = get_tool_spec(env, args.env_name)
        num_trajs = len(trajs)
        traj_ids = np.array_split(np.arange(num_trajs), args.gen_num_batch)[args.gen_batch_id]
        for i in traj_ids:
            tid = tids[i]
            action_mask = tool_spec['action_masks'][tid]
            contact_loss_mask = tool_spec['contact_loss_masks'][tid]
            reset_key = {'init_v': init_vs[i], 'target_v': target_vs[i]}
            print(reset_key)
            reset_key['contact_loss_mask'] = contact_loss_mask
            traj = sample_traj_solver(env, solver, reset_key, tid, action_mask=action_mask, solver_init=args.solver_init, loaded_reset_state=reset_states[i])
            print(
                f"traj {init_vs[i]}, agent time: {traj['info_agent_time']}, env time: {traj['info_env_time']}, total time: {traj['info_total_time']}")
            buffer.add(traj)
            buffer.save(os.path.join(args.dataset_path))
            visualize_dataset(args.dataset_path, env.cfg.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'),
                                    visualize_reset=False,
                                    overlay_target=False,
                                    vis_target=True)
    else:
        if args.data_name == 'demo':
            if isinstance(args.num_trajs, tuple):
                traj_ids = np.array_split(np.arange(*args.num_trajs), args.gen_num_batch)[args.gen_batch_id]
            else:
                traj_ids = np.array_split(np.arange(args.num_trajs), args.gen_num_batch)[args.gen_batch_id]
            tids = list(range(args.num_tools))
            def get_state_goal_id(traj_id):
                if 'CutRearrange' in args.env_name or 'Writer' in args.env_name or 'MultiTool' in args.env_name:
                    np.random.seed(traj_id)
                    state_id = traj_id
                    goal_id = state_id
                else:
                    # Random selection for other env
                    np.random.seed(traj_id)
                    goal_id = np.random.randint(0, env.num_targets)
                    state_id = np.random.randint(0, env.num_inits)
                return {'init_v': state_id, 'target_v': goal_id}  # state and target version
        else:
            tids = [args.tool_combo_id]
            from core.diffskill.hardcoded_eval_trajs import get_eval_traj
            init_vs, target_vs = get_eval_traj(env.cfg.cached_state_path, plan_step = args.plan_step)
            traj_ids = range(len(init_vs))

            def get_state_goal_id(traj_id):
                return {'init_v': init_vs[traj_id], 'target_v': target_vs[traj_id]}  # state and target version

        solver = Solver(args, taichi_env, (0,), return_dist=True)
        args.dataset_path = os.path.join(logger.get_dir(), 'dataset.gz')

        from core.diffskill.env_spec import get_tool_spec
        tool_spec = get_tool_spec(env, args.env_name)    
        for tid in tids:  # Only use the first two tools
            action_mask = tool_spec['action_masks'][tid]
            contact_loss_mask = tool_spec['contact_loss_masks'][tid]
            for i, traj_id in enumerate(traj_ids):
                reset_key = get_state_goal_id(traj_id)
                print(reset_key)
                if args.data_name == 'demo' and not check_correct_tid(args.env_name, reset_key['init_v'], reset_key['target_v'], tid):
                    continue

                reset_key['contact_loss_mask'] = contact_loss_mask
                solver_init = args.solver_init
                traj = sample_traj_solver(env, solver, reset_key, tid, action_mask=action_mask, solver_init=solver_init, loaded_reset_state=None)
                print(
                    f"traj {traj_id}, agent time: {traj['info_agent_time']}, env time: {traj['info_env_time']}, total time: {traj['info_total_time']}")
                buffer.add(traj)
                if i % 10 == 0:
                    buffer.save(os.path.join(args.dataset_path))
                    visualize_dataset(args.dataset_path, env.cfg.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'),
                                    visualize_reset=False,
                                    overlay_target=False,
                                    vis_target=True)
    if args.data_name == 'demo' and args.env_name != 'CutRearrangeSpread-v1' and args.env_name != 'Writer-v1' and args.env_name != 'MultiTool-v1':
        from core.diffskill.generate_reset_motion import generate_reset_motion
        generate_reset_motion(buffer, env)
        buffer.save(os.path.join(args.dataset_path))
        visualize_dataset(args.dataset_path, env.cfg.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'), visualize_reset=True,
                          overlay_target=False, vis_target=True)
    else:
        buffer.save(os.path.join(args.dataset_path))
        visualize_dataset(args.dataset_path, env.cfg.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'), visualize_reset=False,
                          overlay_target=False, vis_target=True)
    if args.use_wandb:
        perf, perf_std = calculate_performance(args.dataset_path, max_step=args.horizon, num_moves=1)
        print("Mean performance so far:", perf)
        wandb.log({'avg performance': perf})
        wandb.log({'std performance': perf_std})