import time
import numpy as np
import torch
import os
from tqdm import tqdm
from chester import logger
import pytorch3d.transforms as transforms
from core.diffskill.env_spec import get_num_traj
from core.diffskill.utils import get_obs_goal_match
from core.toolgen.se3 import random_so3
from core.toolgen.vat_buffer import INIT_TOOL_POSES
from plb.envs.mp_wrapper import SubprocVecEnv
from core.toolgen.bc_agent import BCAgent
from core.toolgen.agent import VatAgent

def get_correct_tid(env_name, init_v, target_v, cur_traj_idx=0):
    if 'Writer' in env_name:
        if init_v < get_num_traj(env_name) // 2:
            correct_tid = 0
        else:
            correct_tid = 1
        return correct_tid
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
            raise NotImplementedError
        return correct_tid
    elif env_name == 'Roll-v1' or env_name == 'Cut-v1' or env_name == 'SmallScoop-v1' or env_name == 'LargeScoop-v1':
        return cur_traj_idx // 10
    else:
        return NotImplementedError

def tid_to_alternative_tid(env_name, tid):
    if env_name =='MultiTool-v1':
        tid_map = {
            0: 1,   # small bowl -> large bowl
            1: 0,
            2: 3,
            3: 2
        }
        return tid_map[tid]
    else:
        return NotImplementedError

def reset_pose_to_alternative_pose(env_name, tid, pose):
    ret_pose = np.copy(pose)
    if env_name =='MultiTool-v1':
        if tid == 0:
            ret_pose[1] += 0.01
        elif tid == 1:
            ret_pose[1] -= 0.01
        return ret_pose
    else:
        return NotImplementedError

def sample_rollout_bc(env, agent, reset_key, buffer=None, cur_traj_idx=0):
    assert agent.args.num_env == 1
    states, obses, actions, rewards, succs, scores = [], [], [], [], [0.], [0.]  # Append 0 for the first frame for succs and scores
    info_tool_select_acc = []
    waypoints = []
    achieved_waypoints = []
    selected_tools = []
    if isinstance(env, SubprocVecEnv):
        if reset_key is not None:
            state = env.reset([reset_key])[0]
        obs = env.render([{'mode': 'rgb', 'img_size': agent.args.img_size}])[0]  # rgbd observation
    else:
        if reset_key is not None:
            state = env.reset(**reset_key)
        # rgbd observation
        obs = env.render(mode='rgb', img_size=agent.args.img_size)
    
    if agent.args.random_init_pose:
        for i in range(agent.args.num_tools):
            cur_tool_state = torch.FloatTensor(state[3000+i*agent.args.dimtool:3000+(i+1)*agent.args.dimtool])[None]
            random_init_rot = random_so3(1, rot_var=np.pi*2).view(1, 4)
            cur_tool_state[:, 3:7] = transforms.quaternion_multiply(random_init_rot, cur_tool_state[:, 3:7])
            env.getfunc(f'taichi_env.primitives[{i}].set_state', 0, list_kwargs=[{'f': 0, 'state': cur_tool_state[0, :7].numpy()}])
        action_dim = env.getattr("taichi_env.primitives.action_dim", 0)
        mp_state, _, _, _ = env.step([np.zeros(action_dim)])
        state = mp_state[0]
        obs = env.render([{'mode': 'rgb', 'img_size': agent.args.img_size}])[0]  # rgbd observation
    if reset_key is not None:
        states.append(state)
        obses.append(obs)
    T = agent.args.horizon
    total_r = 0

    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    action_dim = env.getattr("taichi_env.primitives.action_dim", 0)
    # action_dim = env.taichi_env.primitives.action_dim
    _, _, _, mp_info = env.step([np.zeros(action_dim)])
    # reset init_dist again. This might be important
    env.getfunc('taichi_env.set_init_dist', 0, list_kwargs=[{'loss_type': agent.args.env_loss_type}])
    if reset_key is not None:
        infos = [mp_info[0]]
        # infos = [mp_info]
    else:
        infos = []

    if 'goal_as_flow' in agent.args.actor_type:
        obs = state[:3000].reshape(1000, 3)
        goal = np.array(env.getattr('target_pc', 0)).reshape(1000, 3)
        goal = get_obs_goal_match(obs, goal)
        goal_dpc = torch.FloatTensor(goal[None]).to(agent.device)
    else:
        goal_dpc = torch.FloatTensor(np.array(env.getattr('target_pc', 0))[None]).to(agent.device)
        # goal_dpc = torch.FloatTensor(np.array(env.target_pc)[None]).to(agent.device)

    with torch.no_grad():
            t1 = time.time()
            state = torch.FloatTensor(state).to(agent.device)[None]
            if agent.args.use_gt_tool or agent.args.use_gt_tool_and_reset:
                if agent.args.mask_gt_tool:
                    allowed_tids = [tid_to_alternative_tid(agent.args.env_name, 
                                    get_correct_tid(agent.args.env_name, reset_key['init_v'], reset_key['target_v']))]
                else:
                    allowed_tids = [get_correct_tid(agent.args.env_name, reset_key['init_v'], reset_key['target_v'], cur_traj_idx=cur_traj_idx)]
            elif agent.args.mask_gt_tool:
                allowed_tids = [i for i in range(agent.args.num_tools) if i != get_correct_tid(agent.args.env_name, reset_key['init_v'], reset_key['target_v'])]
            else:
                allowed_tids = range(agent.args.num_tools)
            if not agent.args.use_contact: # BC, taxpose-bc
                if agent.args.opt_reset_pose:
                    pred_traj, tid, opt_plots = agent.act_with_fitting(state, goal_dpc, allowed_tids, save_plots=True)
                elif agent.args.visualize_flow:
                    pred_traj, tid, flow_plots = agent.act(state, goal_dpc, allowed_tids, save_plots=True)
                else:
                    pred_traj, tid = agent.act(state, goal_dpc, allowed_tids, save_plots=False)  # horizon x 6
            else:
                # import pickle
                # with open(os.path.join(agent.args.cached_state_path, 'init/contact_points.pkl'), 'rb') as f:
                #     contact_points = pickle.load(f)
                # contact_point = torch.FloatTensor(contact_points[reset_key['init_v']]).cuda().view(1, 3)
                idx = np.where(buffer.buffer['init_v'] == reset_key['init_v'])[0]
                contact_point = buffer.buffer['contact_points'][idx][0]
                contact_point = torch.FloatTensor(contact_point).cuda().view(1, 3)
                pred_traj, tid = agent.act(state, goal_dpc, contact_point, allowed_tids)

            assert tid in allowed_tids
            pred_traj = pred_traj.detach().cpu()
            if agent.args.use_gt_tool_and_reset:
                assert buffer is not None
                tid = get_correct_tid(agent.args.env_name, reset_key['init_v'], reset_key['target_v'])
                idx = np.where(buffer.buffer['init_v'] == reset_key['init_v'])[0]
                _, tool_reset_pose = buffer.get_state(idx)
                reset_pose = tool_reset_pose.reshape(1, -1, agent.args.dimtool)[0, tid, :7]
                if agent.args.mask_gt_tool: # THIS WILL RESET THE ALTERNATIVE TOOL TO GT TOOL's PROXIMAL RESET POSE
                    tid = tid_to_alternative_tid(agent.args.env_name, tid)
                    reset_pose = reset_pose_to_alternative_pose(agent.args.env_name, tid, reset_pose)
            else:
                pred_reset_dir = transforms.matrix_to_quaternion(transforms.rotation_6d_to_matrix(pred_traj[0])).numpy()
                pred_reset_pos = pred_traj[1, :3].numpy()
                reset_pose = np.concatenate([pred_reset_pos, pred_reset_dir])
            selected_tools.append(tid)
            # check predicted tool
            if agent.args.mask_gt_tool:
                tool_accuracy = int(tid_to_alternative_tid(agent.args.env_name, 
                    get_correct_tid(agent.args.env_name, reset_key['init_v'], reset_key['target_v'])) == tid)
            else:
                tool_accuracy = int(get_correct_tid(agent.args.env_name, reset_key['init_v'], reset_key['target_v'], cur_traj_idx=cur_traj_idx) == tid)
            info_tool_select_acc.append(tool_accuracy)

            env.getfunc(f'taichi_env.primitives[{tid}].set_state', 0, list_kwargs=[{'f': 0, 'state': reset_pose}])
            # env.taichi_env.primitives[tid].set_state(0, reset_pose)
            waypoints.append(reset_pose)
            achieved_waypoints.append(np.concatenate([reset_pose, np.array([0.])]))
            t2 = time.time()
            for i in tqdm(range(2, len(pred_traj)), desc='Eval waypoint step'):
                delta_pos, delta_dir = pred_traj[i, :3].numpy(), pred_traj[i, 3:]
                cur_pose = env.getfunc(f'taichi_env.primitives[{tid}].get_state', 0, list_kwargs=[{'f': 0}])
                n_pos = waypoints[-1][:3] + delta_pos
                n_dir = transforms.quaternion_multiply(
                    transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(delta_dir, "XYZ")), 
                    torch.FloatTensor(cur_pose[3:7])
                    ).numpy()
                n_pose = np.concatenate([n_pos, n_dir])
                # n_pose = np.concatenate([n_pos, waypoints[-1][3:]])
                cur_states, cur_actions, cur_obs, cur_r, cur_infos = env.getfunc('primitive_reset_to', 0,
                                                  list_kwargs=[{
                                                    'idx': tid, 
                                                    'reset_states': [n_pose for _ in range(agent.args.num_tools)], 
                                                    'return_np': False, 
                                                    'img_size': agent.args.img_size
                                                    }])
                # cur_states, cur_actions, cur_obs, cur_r, cur_infos = env.primitive_reset_to(tid, 
                                                    # [n_pose for _ in range(agent.args.num_tools)], 
                                                    # return_np=False, 
                                                    # img_size=agent.args.img_size
                                                    # )]
                if len(cur_states) > 0:
                    print(len(cur_states))
                    achieved_waypoints.append(cur_states[-1][3000+tid*agent.args.dimtool:3000+(tid+1)*agent.args.dimtool])
                waypoints.append(n_pose)
                states.extend(cur_states)
                actions.extend(cur_actions)
                obses.extend(cur_obs)
                rewards.extend(cur_r)
                infos.extend(cur_infos)
            
            t3 = time.time()

            agent_time += t2 - t1
            env_time += t3 - t2

    target_img = np.array(env.getattr('target_img', 0))
    # target_img = np.array(env.target_img)
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
           'waypoints': np.array(waypoints).astype(np.float32),
           'achieved_waypoints': np.array(achieved_waypoints).astype(np.float32),
           'pred_traj': pred_traj.numpy(),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_tool_select_acc': np.array(info_tool_select_acc),
           'selected_tools': np.array(selected_tools),
           'info_emds': emds,
           'info_final_normalized_performance': info_final_normalized_performance,
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time,
    }
    if agent.args.visualize_flow:
        ret['html_flow_plots'] = flow_plots
    if agent.args.opt_reset_pose:
        ret['html_opt_plots'] = opt_plots
    if reset_key is not None:
        ret.update(**reset_key)
    return ret

def sample_rollout_buffer(env, args, buffer, reset_key):
    states, obses, actions, rewards = [], [], [], [] # Append 0 for the first frame for succs and scores
    if isinstance(env, SubprocVecEnv):
        if reset_key is not None:
            state = env.reset([reset_key])[0]
        obs = env.render([{'mode': 'rgb', 'img_size': args.img_size}])[0]  # rgbd observation
    else:
        if reset_key is not None:
            state = env.reset(**reset_key)
        # rgbd observation
        obs = env.render(mode='rgb', img_size=args.img_size)

    if reset_key is not None:
        states.append(state)
        obses.append(obs)
    T = args.horizon
    total_r = 0

    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    action_dim = env.getattr("taichi_env.primitives.action_dim", 0)
    # action_dim = env.taichi_env.primitives.action_dim
    _, _, _, mp_info = env.step([np.zeros(action_dim)])
    # reset init_dist again. This might be important
    env.getfunc('taichi_env.set_init_dist', 0, list_kwargs=[{'loss_type': args.env_loss_type}])
    if reset_key is not None:
        infos = [mp_info[0]]
        # infos = [mp_info]
    else:
        infos = []

    state = torch.FloatTensor(state).to('cuda')[None]
    tid = get_correct_tid(args.env_name, reset_key['init_v'], reset_key['target_v'])
    idx = np.where(buffer.buffer['init_v'] == reset_key['init_v'])[0]
    _, tool_reset_pose = buffer.get_state(idx)
    reset_pose = tool_reset_pose.reshape(1, -1, args.dimtool)[0, tid, :7]
    env.getfunc(f'taichi_env.primitives[{tid}].set_state', 0, list_kwargs=[{'f': 0, 'state': reset_pose}])
    action_sequence = buffer.buffer['actions'][idx[0]:idx[0]+buffer.horizon]
    for i in range(len(action_sequence)):
        t1 = time.time()

        with torch.no_grad():
            action = action_sequence[i]
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
            [{'mode': 'rgb', 'img_size': args.img_size}])[0]
        state = next_state
        obses.append(obs)
        total_r += reward
        rewards.append(reward)

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
    if reset_key is not None:
        ret.update(**reset_key)
    return ret


def sample_rollout_traj(env, args, traj, tid):
    states, obses, actions, rewards = [], [], [], [] # Append 0 for the first frame for succs and scores
    reset_key = {'init_v': traj['init_v'], 'target_v':traj['target_v']}
    state = env.reset([reset_key])[0]
    env.getfunc(f'taichi_env.primitives[{0}].set_state', 0, list_kwargs=[{'f': 0, 'state': [0.1, 0.9, 0.1, 1, 0, 0, 0]}])
    env.getfunc(f'taichi_env.primitives[{1}].set_state', 0, list_kwargs=[{'f': 0, 'state': [0.1, 0.9, 0.1, 1, 0, 0, 0]}])
    env.getfunc(f'taichi_env.primitives[{2}].set_state', 0, list_kwargs=[{'f': 0, 'state': [0.1, 0.9, 0.1, 1, 0, 0, 0]}])
    obs = env.render([{'mode': 'rgb', 'img_size': args.img_size}])[0]  # rgbd observation
    states.append(state)
    obses.append(obs)
    T = args.horizon
    total_r = 0

    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    action_dim = env.getattr("taichi_env.primitives.action_dim", 0)
    # action_dim = env.taichi_env.primitives.action_dim
    _, _, _, mp_info = env.step([np.zeros(action_dim)])
    # reset init_dist again. This might be important
    env.getfunc('taichi_env.set_init_dist', 0, list_kwargs=[{'loss_type': args.env_loss_type}])
    if reset_key is not None:
        infos = [mp_info[0]]
        # infos = [mp_info]
    else:
        infos = []

    state = torch.FloatTensor(state).to('cuda')[None]
    pred_traj = torch.FloatTensor(traj['pred_traj'])
    pred_reset_dir = transforms.matrix_to_quaternion(transforms.rotation_6d_to_matrix(pred_traj[0])).numpy()
    pred_reset_pos = pred_traj[1, :3].numpy()
    reset_pose = np.concatenate([pred_reset_pos, pred_reset_dir])
    env.getfunc(f'taichi_env.primitives[{tid}].set_state', 0, list_kwargs=[{'f': 0, 'state': reset_pose}])
    if True:
        action_sequence = traj['actions']
    else:
        action_sequence = traj['actions'][:, tid*6:(tid+1)*6]
    
    for i in range(len(action_sequence)):
        t1 = time.time()

        with torch.no_grad():
            action = action_sequence[i]
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
            [{'mode': 'rgb', 'img_size': args.img_size}])[0]
        state = next_state
        obses.append(obs)
        total_r += reward
        rewards.append(reward)

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
    if reset_key is not None:
        ret.update(**reset_key)
    return ret