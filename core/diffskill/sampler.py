import time
import numpy as np
import torch
import os
import pickle
from chester import logger

from core.diffskill.utils import get_camera_matrix, get_img, get_partial_pcl2, img_to_tensor, to_action_mask, batch_pred, LIGHT_DOUGH, DARK_DOUGH, LIGHT_TOOL, DARK_TOOL
from core.diffskill.env_spec import get_reset_tool_state
from core.diffskill.utils import img_to_tensor, to_action_mask, batch_pred
from core.utils.pc_utils import decompose_pc, resample_pc
from plb.envs.mp_wrapper import SubprocVecEnv

device = 'cuda'


def sample_traj_agent(env, agent, reset_key, tid, log_succ_score=False, reset_primitive=False):
    """Compute ious: pairwise iou between each pair of timesteps. """
    assert agent.args.num_env == 1
    states, obses, actions, rewards, succs, scores = [], [], [], [], [0.], [0.]  # Append 0 for the first frame for succs and scores

    if isinstance(env, SubprocVecEnv):
        if reset_key is not None:
            state = env.reset([reset_key])[0]
        obs = env.render([{'mode': 'rgb', 'img_size': agent.args.img_size}])[0]  # rgbd observation
    else:
        if reset_key is not None:
            state = env.reset(**reset_key)
        # rgbd observation
        obs = env.render(mode='rgb', img_size=agent.args.img_size)

    if reset_key is not None:
        states.append(state)
        obses.append(obs)
    T = 50
    total_r = 0

    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    action_dim = env.getattr("taichi_env.primitives.action_dim", 0)
    frame_stack = agent.args.frame_stack
    _, _, _, mp_info = env.step([np.zeros(action_dim)])
    if reset_primitive:
        primitive_state = env.getfunc('get_primitive_state', 0)
    if reset_key is not None:
        infos = [mp_info[0]]
    else:
        infos = []

    if agent.args.input_mode == 'rgbd':
        stack_obs = img_to_tensor(np.array(obs)[None], mode=agent.args.img_mode).to(agent.device)  # stack_obs shape: [1, 4, 64, 64]
        target_img = img_to_tensor(np.array(env.getattr('target_img', 0))[None], mode=agent.args.img_mode).to(agent.device)
        C = stack_obs.shape[1]
        stack_obs = stack_obs.repeat([1, frame_stack, 1, 1])
    else:
        goal_dpc = torch.FloatTensor(np.array(env.getattr('target_pc', 0))[None]).to(agent.device)

    with torch.no_grad():
        for i in range(T):
            t1 = time.time()
            with torch.no_grad():
                if agent.args.input_mode == 'rgbd':
                    obs_tensor = img_to_tensor(
                        np.array(obs)[None], mode=agent.args.img_mode).to(agent.device)
                    stack_obs = torch.cat([stack_obs, obs_tensor], dim=1)[:, -frame_stack * C:]
                    action, done, _ = agent.act_rgbd(stack_obs, target_img, tid)
                    if log_succ_score:
                        z_obs, _, _ = agent.vae.encode(stack_obs)
                        z_goal, _, _ = agent.vae.encode(target_img)
                        if i == 0:
                            z_init = z_obs.clone()
                        succ = batch_pred(agent.feas[tid], {
                            'obs': z_init, 'goal': z_obs, 'eval': True}).detach().cpu().numpy()[0]
                        score = batch_pred(agent.reward_predictor, {
                            'obs': z_obs, 'goal': z_goal, 'eval': True}).detach().cpu().numpy()[0]
                else:
                    state = torch.FloatTensor(state).to(agent.device)[None]
                    action, done, info = agent.act(state, goal_dpc, tid)
                    if log_succ_score:
                        u_obs, u_goal = info['u_obs'], info['u_goal']
                        if i == 0:
                            u_init = u_obs.clone()
                        succ = batch_pred(agent.feas[tid], {
                            'obs': u_init, 'goal': u_obs, 'eval': True}).detach().cpu().numpy()[0]
                        score = batch_pred(agent.reward_predictor, {
                            'obs': u_obs, 'goal': u_goal, 'eval': True}).detach().cpu().numpy()[0]
            action = action[0].detach().cpu().numpy()
            done = done[0].detach().cpu().numpy()
            if np.round(done).astype(int) == 1 and agent.terminate_early:
                break
            t2 = time.time()
            mp_next_state, mp_reward, _, mp_info = env.step([action])
            next_state, reward, info = mp_next_state[0], mp_reward[0], mp_info[0]

            infos.append(info)
            t3 = time.time()

            agent_time += t2 - t1
            env_time += t3 - t2

            actions.append(action)
            states.append(next_state)
            obs = env.render(
                [{'mode': 'rgb', 'img_size': agent.args.img_size}])[0]
            state = next_state
            obses.append(obs)
            total_r += reward
            rewards.append(reward)
            if log_succ_score:
                succs.append(succ)
                scores.append(score)
        if reset_primitive:
            _, _, reset_obses, _, _ = env.getfunc('primitive_reset_to', 0,
                                                  list_kwargs=[{'idx': tid, 'reset_states': primitive_state}])  # TODO tid
            for obs in reset_obses:
                assert frame_stack == 1
                if log_succ_score:
                    if agent.args.input_mode == 'rgbd':
                        with torch.no_grad():
                            z_obs, _, _ = agent.vae.encode(stack_obs)
                            z_goal, _, _ = agent.vae.encode(target_img)
                            if i == 0:
                                z_init = z_obs.clone()
                            succ = batch_pred(agent.feas[tid], {
                                'obs': z_init, 'goal': z_obs, 'eval': True}).detach().cpu().numpy()[0]
                            score = batch_pred(agent.reward_predictor, {
                                'obs': z_obs, 'goal': z_goal, 'eval': True}).detach().cpu().numpy()[0]
                    else:
                        succ = 0
                        score = 0
                obses.append(obs)
                if log_succ_score:
                    succs.append(succ)
                    scores.append(score)

    target_img = np.array(env.getattr('target_img', 0))
    emds = np.array([info['info_emd'] for info in infos])
    if len(infos) > 0:
        info_normalized_performance = np.array(
            [info['info_normalized_performance'] for info in infos])
        info_final_normalized_performance = info_normalized_performance[-1]
    else:
        info_normalized_performance = []
        info_final_normalized_performance = None

    total_time = time.time() - st_time
    ret = {'states': np.array(states).astype(np.float32),
           'obses': np.array(obses).astype(np.float32),
           'actions': np.array(actions).astype(np.float32),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_emds': emds,
           'info_final_normalized_performance': info_final_normalized_performance,
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time}
    if log_succ_score:
        ret['succs'] = np.array(succs)  # Should miss the first frame
        ret['scores'] = np.array(scores)
    if reset_key is not None:
        ret.update(**reset_key)
    return ret


def sample_traj_solver(env, agent, reset_key, tid, action_mask=None, reset_primitive=False, primitive_reset_states=None, solver_init='zero',
                       num_moves=1, loaded_reset_state=None):
    """Compute ious: pairwise iou between each pair of timesteps. """
    assert not isinstance(env, SubprocVecEnv)
    states, obses, actions, rewards = [], [], [], []
    contact_points = []
    if action_mask is None:
        if tid == 0:
            action_mask = to_action_mask(env, [1, 0])
        else:
            action_mask = to_action_mask(env, [0, 1])
    if reset_key is not None:
        state = env.reset(**reset_key)
        contact_point = env.taichi_env.primitives[tid].get_contact_point(0, state[:3000].reshape(1000, 3)).reshape(1, 3)
    if reset_primitive:
        if primitive_reset_states is None:
            primitive_reset_states = []
            primitive_reset_states.append(env.get_primitive_state())
        for idx, prim in enumerate(env.taichi_env.primitives):
            prim.set_state(0, primitive_reset_states[0][idx])

    # rgbd observation
    obs = env.render(mode='rgb', img_size=agent.args.img_size)

    if reset_key is not None:
        states.append(state)
        obses.append(obs)
        contact_points.append(contact_point)
    T = 50
    total_r = 0
    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()

    # Solver
    taichi_env = env.taichi_env
    action_dim = taichi_env.primitives.action_dim
    for _ in range(10):
        state, _, _, info = env.step(np.zeros(action_dim))

    if agent.args.env_name == 'Writer-v1' or agent.args.env_name == 'MultiTool-v1':
        if loaded_reset_state is not None:
            env.taichi_env.primitives[tid].set_state(0, loaded_reset_state)
        else:
            path = os.path.join(env.cfg.cached_state_path, 'init/init_tool_states.pkl')
            print(path)
            with open(path, 'rb') as f:
                tool_states = pickle.load(f)
            tool_reset_state = tool_states[reset_key['init_v']]
            env.taichi_env.primitives[tid].set_state(0, tool_reset_state)

    infos = [info]
    actions = []
    for move in range(num_moves):
        if solver_init == 'multiple':
            init_actions = np.zeros(
                [3, T, taichi_env.primitives.action_dim], dtype=np.float32)  # left, right, zero
            init_actions[0, :10, 3] = -1.  # left
            init_actions[1, :10, 3] = 1.  # right
            all_infos = []
            all_buffers = []
            for i in range(len(init_actions)):
                init_action = init_actions[i]
                cur_info, cur_buffer = agent.solve(init_action, action_mask=action_mask, loss_fn=taichi_env.compute_loss,
                                                   max_iter=agent.args.gd_max_iter,
                                                   lr=agent.args.lr)
                all_infos.append(cur_info)
                all_buffers.append(cur_buffer)
            improvements = np.array(
                [(all_buffers[i][0]['loss'] - all_infos[i]['best_loss']) / all_buffers[i][0]['loss'] for i in range(len(all_infos))])
            solver_info = all_infos[np.argmax(improvements)]
        else:
            if isinstance(solver_init, float):
                cut_loc = solver_init
                init_action = np.zeros(
                    [T, taichi_env.primitives.action_dim], dtype=np.float32)
                cur_loc = state[3000]
                act_x = (cut_loc - cur_loc) / 0.015 / 10
                init_action[:10, 0] = act_x
                init_action[10:20, 1] = -1
            elif solver_init == 'zero':
                init_action = np.zeros(
                    [T, taichi_env.primitives.action_dim], dtype=np.float32)
            elif solver_init == 'normal':
                init_action = np.random.normal(
                    size=(T, taichi_env.primitives.action_dim))
            elif solver_init == 'uniform':
                init_action = np.random.uniform(-0.2, 0.2, size=(T, taichi_env.primitives.action_dim))
            elif solver_init == 'left':
                init_action = np.zeros(
                    [T, taichi_env.primitives.action_dim], dtype=np.float32)
                init_action[:10, 3] = -1 * np.ones((10,), dtype=np.float32)
                print(init_action)
            else:
                raise NotImplementedError
            
            max_iter = agent.args.gd_max_iter
            if agent.args.env_name == 'MultiTool-v1' and tid == 4:
                init_action[:10, 6*tid+1] = -1
                print("first action:", init_action[:10])
            elif agent.args.env_name == 'MultiTool-v1' and reset_key['init_v'] >= 800:
                path = os.path.join(env.cfg.cached_state_path, 'init/traj_actions.pkl')
                with open(path, 'rb') as f:
                    traj_actions = pickle.load(f)   
                traj_action = np.array(traj_actions[reset_key['init_v']])
                if reset_key['init_v'] >= 1000:
                    tid_actions = traj_action[:, 6:12].copy()
                    traj_action[:, :6] = tid_actions
                else:
                    tid_actions = traj_action[:, :6].copy()
                    traj_action[:, 6:12] = tid_actions
                init_action[:len(traj_action)] = traj_action
                max_iter = 0
            solver_info, _ = agent.solve(init_action, action_mask=action_mask, loss_fn=taichi_env.compute_loss, max_iter=max_iter,
                                         lr=agent.args.lr)

        agent.save_plot_buffer(os.path.join(logger.get_dir(), f'solver_loss.png'))
        agent.dump_buffer(os.path.join(logger.get_dir(), f'buffer.pkl'))
        solver_actions = solver_info['best_action']
        actions.extend(solver_actions)
        agent_time = time.time() - st_time
        for i in range(T):
            t1 = time.time()
            next_state, reward, _, info = env.step(solver_actions[i])
            contact_point = env.taichi_env.primitives[tid].get_contact_point(0, next_state[:3000].reshape(1000, 3)).reshape(1, 3)
            contact_points.append(contact_point)
            infos.append(info)
            env_time += time.time() - t1
            states.append(next_state)
            obs = taichi_env.render(mode='rgb', img_size=agent.args.img_size)
            obses.append(obs)
            total_r += reward
            rewards.append(reward)
            # mass_grids.append(info['mass_grid'])
        target_img = env.target_img

        if reset_primitive and move < num_moves - 1:
            if len(primitive_reset_states) < num_moves + 1:
                primitive_reset_states.append(primitive_reset_states[-1])
            else:
                assert len(primitive_reset_states) == num_moves + 1
            for idx, prim in enumerate(env.taichi_env.primitives):
                prim.set_state(0, primitive_reset_states[move + 1][idx])

        emds = np.array([info['info_emd'] for info in infos])
        if len(infos) > 0:
            info_normalized_performance = np.array(
                [info['info_normalized_performance'] for info in infos])
            info_final_normalized_performance = info_normalized_performance[-1]
        else:
            info_normalized_performance = []
            info_final_normalized_performance = None

    total_time = time.time() - st_time
    ret = {'states': np.array(states).astype(np.float32),
           'contact_points': np.array(contact_points).astype(np.float32),
           'obses': np.array(obses).astype(np.float32),
           'actions': np.array(actions).astype(np.float32),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_emds': emds,
           'info_final_normalized_performance': info_final_normalized_performance,
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time,
           'action_mask': action_mask}
    if reset_key is not None:
        ret.update(**reset_key)
    return ret


def sample_traj_replay(env, reset_key, tid, action_sequence, action_mask=None, img_size=64, save_mode=None, args=None, init_states=None):
    """Compute ious: pairwise iou between each pair of timesteps. """
    def overlay(img1, img2):
        mask = img2[:, :, 3][:, :, None]
        return img1 * (1 - mask) + img2 * mask

    assert not isinstance(env, SubprocVecEnv)
    states, obses, actions, rewards = [], [], [], []
    if action_mask is None:
        if tid == 0:
            action_mask = to_action_mask(env, [1, 0])
        else:
            action_mask = to_action_mask(env, [0, 1])

    if isinstance(env, SubprocVecEnv):
        if reset_key is not None:
            state = env.reset([reset_key])[0]
        obs = env.render([{'mode': 'rgb', 'img_size': img_size}])[
            0]  # rgbd observation
    else:
        if reset_key is not None:
            state = env.reset(**reset_key)
        obs = env.render(mode='rgb', img_size=img_size)  # rgbd observation

    if reset_key is not None:
        states.append(state)
        obses.append(obs)
    total_r = 0
    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    # Replay trajectories
    action_dim = env.taichi_env.primitives.action_dim
    _, _, _, info = env.step(np.zeros(action_dim))
    infos = [info]
    xs = np.linspace(0., 1., len(action_sequence))
    ys = []

    if init_states is not None:
        assert len(init_states) == len(env.taichi_env.primitives)
        for idx, prim in enumerate(env.taichi_env.primitives):
            prim.set_state(0, init_states[idx])
    for i in range(len(action_sequence)):
        t1 = time.time()

        with torch.no_grad():
            action = action_sequence[i]
        t2 = time.time()
        next_state, reward, _, info = env.step(action)
        t3 = time.time()

        agent_time += t2 - t1
        env_time += t3 - t2

        actions.append(action)
        states.append(next_state)
        obs = env.render(mode='rgb', img_size=img_size)
        if save_mode=='plot':
            ys.append(info['info_normalized_performance'])
            img = get_img(args, xs[:len(ys)], ys)
            obs = np.clip(overlay(obs, img), 0., 1.)
        obses.append(obs)
        infos.append(info)
        total_r += reward
        rewards.append(reward)
    target_img = env.target_img
    emds = np.array([info['info_emd'] for info in infos])
    if len(infos) > 0:
        info_normalized_performance = np.array(
            [info['info_normalized_performance'] for info in infos])
        info_final_normalized_performance = info_normalized_performance[-1]
    else:
        info_normalized_performance = []
        info_final_normalized_performance = None

    total_time = time.time() - st_time
    ret = {'states': np.array(states).astype(np.float32),
           'obses': np.array(obses).astype(np.float32),
           'actions': np.array(actions).astype(np.float32),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_emds': emds,
           'info_final_normalized_performance': info_final_normalized_performance,
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time,
           'action_mask': action_mask}
    if reset_key is not None:
        ret.update(**reset_key)
    return ret


def sample_traj(env, agent, reset_key, tid, action_mask=None, action_sequence=None, log_succ_score=False, reset_primitive=False, solver_init='zero',
                num_moves=1):
    print("This function is no longer being used. Please call one of `sample_traj_agent`, `sample_traj_solver`, or `sample_traj_reply` instead.")
    raise NotImplementedError
