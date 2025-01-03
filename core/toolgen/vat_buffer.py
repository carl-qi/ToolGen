import torch
import numpy as np
import os
from core.diffskill.utils import batch_rand_int, img_to_tensor, img_to_np
from core.diffskill.buffer import ReplayBuffer
from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms as transforms
from core.diffskill.env_spec import get_reset_tool_state
from core.toolgen.se3 import random_so3

INIT_TOOL_POSES = {
    0: np.array([0.7071, 0.7071, 0., 0.]),
    1: np.array([0.7071, 0.7071, 0., 0.]),
    2: np.array([0., 0., 0., 1.]),
    3: np.array([0., 0., 0., 1.]),
    4: np.array([1., 0., 0., 0.]),
}

def filter_buffer_nan(buffer):
    actions = buffer.buffer['actions']
    idx = np.where(np.isnan(actions))
    print('{} nan actions detected. making them zero.'.format(len(idx[0])))
    buffer.buffer['actions'][idx] = 0.


class VatMartDataset(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(VatMartDataset, self).__init__(*args, **kwargs)
        self.tool_traj_idxes = {}
        self.tool_idxes = {}
        self.stats = {}

    # Should also be consistent with the one when generating the feasibility prediction dataset! line 291 of train.py
    def get_tid(self, action_mask):
        """ Should be consistent with get_tool_idxes"""
        if len(action_mask.shape) == 1:
            return int(action_mask[0] < 0.5)
        elif len(action_mask.shape) == 2:
            return np.array(action_mask[:, 0] < 0.5).astype(np.int)
    
    def compute_stats(self):
        all_pc = self.buffer['states'][:self.cur_size, :3000].reshape(-1, 1000, 3)
        all_pc = np.vstack([all_pc, self.np_target_pc])
        centered_pc = all_pc - np.mean(all_pc, axis=1, keepdims=True)
        std = np.std(centered_pc.flatten())
        self.stats['std'] = std
        print("buffer stats:", self.stats)
        
    def get_epoch_tool_idx(self, epoch, tid, mode):
        # Get a index generator for each tool index of the mini-batch
        # Note: Assume static buffer
        assert mode in ['train', 'eval']
        cache_name = f'{mode}_{tid}'
        if cache_name not in self.tool_idxes:
            idxes = self.train_idx if mode == 'train' else self.eval_idx
            tool_idxes = self.get_tool_idxes(tid, idxes)
            # Shuffle the tool idxes so that if the trajectories in the replay buffer is ordered, this will shuffle the order.
            tool_idxes = tool_idxes.reshape(-1, 50)
            perm = np.random.permutation(len(tool_idxes))
            tool_idxes = tool_idxes[perm].flatten()
            self.tool_idxes[cache_name] = tool_idxes
        if mode == 'train':
            num_steps = len(self.tool_idxes[cache_name])
            B = self.args.batch_size
        else:
            num_steps = len(self.tool_idxes[cache_name])
            B = self.args.batch_size 
        permuted_idx = self.tool_idxes[cache_name][np.random.permutation(num_steps)]
        epoch_tool_idxes = [permuted_idx[np.arange(i, min(i + B, num_steps))] for i in range(0, num_steps, B)]
        return epoch_tool_idxes
    
    def get_epoch_tool_traj_idx(self, epoch, tid, mode):
        assert mode in ['train', 'eval']
        cache_name = f'{mode}_{tid}'
        if cache_name not in self.tool_traj_idxes:
            traj_idxes = self.train_traj_idx if mode == 'train' else self.eval_traj_idx
            tool_traj_idxes = self.get_tool_idxes(tid, traj_idxes * self.horizon) // self.horizon
            perm = np.random.permutation(len(tool_traj_idxes))
            tool_traj_idxes = tool_traj_idxes[perm].flatten()
            self.tool_traj_idxes[cache_name] = tool_traj_idxes
        num_steps = len(self.tool_traj_idxes[cache_name])
        B = self.args.batch_size
        permuted_idx = self.tool_traj_idxes[cache_name][np.random.permutation(num_steps)]
        epoch_tool_traj_idxes = [permuted_idx[np.arange(i, min(i + B, num_steps))] for i in range(0, num_steps, B)]
        return epoch_tool_traj_idxes

    def sample_goal(self, obs_idx, hindsight_goal_ratio, device):
        n = len(obs_idx)
        horizon = 50
        traj_id = obs_idx // horizon
        traj_t = obs_idx % horizon
        init_v, target_v = self.buffer['init_v'][obs_idx], self.buffer['target_v'][obs_idx]
        hindsight_future_idx = batch_rand_int(traj_t, horizon, n)
        optimal_step = self.get_optimal_step()
        hindsight_flag = np.round(np.random.random(n) < hindsight_goal_ratio).astype(np.int32)
        if self.args.input_mode == 'rgbd':
            hindsight_goal_imgs = self.buffer['obses'][hindsight_future_idx + traj_id * horizon]
            target_goal_imgs = self.np_target_imgs[target_v]
            goal_obs = img_to_tensor(hindsight_flag[:, None, None, None] * hindsight_goal_imgs +
                                     (1 - hindsight_flag[:, None, None, None]) * target_goal_imgs, mode=self.args.img_mode).to(device,
                                                                                                                               non_blocking=True)
        elif self.args.input_mode == 'pc':
            hindsight_goal_states = self.buffer['states'][hindsight_future_idx + traj_id * horizon]
            hindsight_goal_states = hindsight_goal_states[:, :3000].reshape(n, -1, 3)
            target_goal_grid = self.np_target_pc[target_v]
            goal_obs = torch.FloatTensor(hindsight_flag[:, None, None] * hindsight_goal_states +  # replace target goal imgs by target point cloud
                                         (1 - hindsight_flag[:, None, None]) * target_goal_grid).to(device, non_blocking=True)
        else:
            raise NotImplementedError
        hindsight_done_flag = (hindsight_future_idx == 0).astype(np.int32)
        target_done_flag = (traj_t == optimal_step[traj_id, traj_t]).astype(np.int32)
        done_flag = hindsight_flag * hindsight_done_flag + (1 - hindsight_flag) * target_done_flag
        done_flag = torch.FloatTensor(done_flag).to(device, non_blocking=True)
        hindsight_flag = torch.FloatTensor(hindsight_flag).to(device, non_blocking=True)
        return goal_obs, done_flag, hindsight_flag

    def sample_positive_idx(self, obs_idx):
        n = len(obs_idx)
        horizon = 50
        traj_id = obs_idx // horizon
        traj_t = obs_idx % horizon
        future_idx = batch_rand_int(traj_t, horizon, n) + traj_id * horizon
        return future_idx

    def sample_negative_idx(self, obs_idx, epoch):
        horizon = 50
        # Number of trajectories used in the first few epochs, which can be used for negative sampling
        num_traj = min(self.args.step_per_epoch * (epoch + 1) + self.args.step_warmup, len(self)) // horizon
        assert num_traj > 1
        n = len(obs_idx)
        traj_id = obs_idx // horizon
        neg_traj_id = (traj_id + np.random.randint(1, num_traj)) % num_traj
        traj_t = np.random.randint(0, horizon, n)
        neg_idx = neg_traj_id * horizon + traj_t
        return neg_idx

    def sample_reset_obs(self, obs_idx):
        horizon = 50
        traj_id = obs_idx // horizon
        reset_lens = self.buffer['reset_motion_lens'][traj_id]

        did_reset_idx = np.where(reset_lens > 0)
        traj_id = traj_id[did_reset_idx]
        reset_lens = reset_lens[did_reset_idx]
        reset_idx = batch_rand_int(0, reset_lens, len(reset_lens))
        reset_imgs = self.buffer['reset_motion_obses'][traj_id, reset_idx]
        reset_states = self.buffer['reset_state'][traj_id, reset_idx]
        return did_reset_idx, reset_imgs, reset_states


    def get_state(self, idxes):
        num_primitives = self.args.num_tools
        if self.args.input_mode == 'pc':
            dpc = self.buffer['states'][idxes, :3000].reshape(-1, 1000, 3)
        else:
            dpc = self.buffer['dough_pcl'][idxes].reshape(-1, 1000, 3)
        tool_state = self.buffer['states'][idxes, 3000:3000+num_primitives*self.args.dimtool].reshape(len(idxes), -1)

        return dpc, tool_state

    def get_tool_particles(self, idxes, tid):
        n = self.buffer['states'].shape[1]
        assert n > 3300
        num_primitives = self.args.num_tools
        tool_particle_idx = 3000 + num_primitives*self.args.dimtool + tid*300
        tool_particles = self.buffer['states'][idxes, tool_particle_idx:tool_particle_idx+300].reshape(-1, 100, 3)
        return tool_particles

    def tool_traj_to_delta(self, tool_traj, contact_points=None):
        # Assume that tool_traj is B x horizon x (3D rotation + 4D quaternion wxyz)
        # tool_traj[:, 0, :] is the beginning pose
        # tool_traj[:, 1, :] is the reset pose
        # Returns waypoints B x horizon x 6, where waypoint[:, 0, :] is the reset rotation in 6D representation, 
        # waypoint[:, 1, :] is the reset position relative to contact points, if contactpoint is not None
        B = tool_traj.shape[0]

        init_rot = transforms.quaternion_to_matrix(tool_traj[:, 1, 3:7])
        init_rot = transforms.matrix_to_rotation_6d(init_rot).view(B, 1, 6)  # B x 1 x 6

        if contact_points is not None:
            init_pos = (tool_traj[:, 1, :3] - contact_points).view(B, 1, 3)
        else:
            init_pos = tool_traj[:, 1, :3].view(B, 1, 3)
        init_pos = init_pos.repeat(1, 1, 2)  # B x 1 x 6

        delta_pos = tool_traj[:, 2:, :3] - tool_traj[:, 1:-1, :3]  # B x horizon-2 x 3
        rotations_n = tool_traj[:, 2:, 3:7]
        rotations_inv = transforms.quaternion_invert(tool_traj[:, 1:-1, 3:7])
        delta_rotations = transforms.quaternion_multiply(rotations_n, rotations_inv)
        delta_angles = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(delta_rotations), 'XYZ')  # B x horizon-2 x 3
        delta_waypoints = torch.cat([delta_pos, delta_angles], dim=-1)

        return torch.cat([init_rot, init_pos, delta_waypoints], dim=1).reshape(B, -1)

    def sample_transition_implicit(self, batch_traj_idxes, device):
        ret = {}
        for key in ['obses', 'goal_obses', 'tool_reset_states', 'tool_rand_states']:
            ret[key] = []
        for tid, curr_tool_traj_idx in zip(self.args.train_tool_idxes, batch_traj_idxes):
            b = self.args.implicit_n_poses
            B = self.args.batch_size
            batch_idxes = self.horizon * curr_tool_traj_idx
            obs, tool_reset_states = self.get_state(batch_idxes)
            tool_reset_states = tool_reset_states.reshape(B, -1, self.args.dimtool)[:, tid, :]

            target_v = self.buffer['target_v'][batch_idxes]
            goal_obs = self.np_target_pc[target_v]
            obs = torch.FloatTensor(obs).to(device, non_blocking=True)
            goal_obs = torch.FloatTensor(goal_obs).to(device, non_blocking=True)
            tool_reset_states = torch.FloatTensor(tool_reset_states).to(device, non_blocking=True)
            
            # Sample a random rotation
            random_quats = random_so3(b, rot_var=np.pi*2).to(device, non_blocking=True)
            random_trans = torch.FloatTensor(np.concatenate([np.random.uniform(0.45, 0.55, (b, 1)), 
                                                            np.random.uniform(0.1, 0.25, (b, 1)), 
                                                            np.random.uniform(0.45, 0.55, (b, 1))], axis=-1)).to(device, non_blocking=True)
            rand_tool_states = torch.cat([random_trans, random_quats], dim=-1)
            ret['obses'].append(obs)
            ret['goal_obses'].append(goal_obs)
            ret['tool_rand_states'].append(rand_tool_states)
            ret['tool_reset_states'].append(tool_reset_states)
        ret['stats'] = self.stats
        return ret

    def sample_transition_taxpose(self, batch_traj_idxes, device):
        ret = {}
        for key in ['obses', 'goal_obses', 'pos_init_tool_pose', 'pos_reset_tool_pose', 'R_gt', 't_gt','gt_T_action']:
            ret[key] = []
        for tid, curr_tool_traj_idx in zip(self.args.train_tool_idxes, batch_traj_idxes):
            b = self.args.taxpose_n_poses
            batch_idxes = self.horizon * curr_tool_traj_idx
            obs, tool_reset_states = self.get_state(batch_idxes)
            obs = np.tile(obs, (b, 1, 1))
            tool_reset_states = np.tile(tool_reset_states, (b, 1))
            tool_reset_states = tool_reset_states.reshape(b, -1, self.args.dimtool)[:, tid, :]

            target_v = self.buffer['target_v'][batch_idxes]
            goal_obs = self.np_target_pc[target_v]
            goal_obs = np.tile(goal_obs, (b, 1, 1))
            obs = torch.FloatTensor(obs).to(device, non_blocking=True)
            goal_obs = torch.FloatTensor(goal_obs).to(device, non_blocking=True)
            tool_reset_states = torch.FloatTensor(tool_reset_states).to(device, non_blocking=True)
            
            # Sample a random rotation
            random_quats = random_so3(b, rot_var=np.pi*2).to(device, non_blocking=True)
            random_trans = torch.FloatTensor(np.concatenate([np.random.uniform(0.1, 0.9, (b, 1)), 
                                                            np.random.uniform(0.05, 0.25, (b, 1)), 
                                                            np.random.uniform(0.1, 0.9, (b, 1))], axis=-1)).to(device, non_blocking=True)
            tool_init_poses = torch.cat([random_trans, random_quats], dim=-1)
            # Computing transformations
            R_gt = transforms.quaternion_multiply(tool_reset_states[:, 3:7], transforms.quaternion_invert(tool_init_poses[:, 3:7]))
            t_gt = tool_reset_states[:, :3] - transforms.quaternion_apply(R_gt, tool_init_poses[:, :3])
            R_gt = transforms.quaternion_to_matrix(R_gt)
            mat = torch.zeros(b, 4, 4).to(device)
            mat[:, :3, :3] = R_gt
            mat[:, :3, 3] = t_gt
            mat[:, 3, 3] = 1
            gt_T_action = transforms.Transform3d(
                    device=device, matrix=mat.transpose(-1, -2)
            )
            ret['obses'].append(obs)
            ret['goal_obses'].append(goal_obs)
            ret['pos_init_tool_pose'].append(tool_init_poses)
            ret['pos_reset_tool_pose'].append(tool_reset_states)
            ret['R_gt'].append(R_gt)
            ret['t_gt'].append(t_gt)
            ret['gt_T_action'].append(gt_T_action)
        ret['stats'] = self.stats
        return ret
            
    def sample_transition_openloop(self, batch_traj_idxes, device, use_contact=True):
        ret = {}
        for key in ['obses', 'goal_obses', 'contact_points', 'pos_tool_trajs', 'neg_tool_trajs', 'success_flag', 'matched_goal_obses']:
            ret[key] = []
        for tid, curr_tool_traj_idx in zip(self.args.train_tool_idxes, batch_traj_idxes):
            b = curr_tool_traj_idx.shape[0]
            batch_idxes = self.horizon * curr_tool_traj_idx
            
            obs, _ = self.get_state(batch_idxes)
            target_v = self.buffer['target_v'][batch_idxes]
            goal_obs = self.np_target_pc[target_v]
            if self.args.actor_type == 'conditioned_goal_as_flow':
                matched_goal_obs = self.matched_goals[target_v]
                matched_goal_obs = torch.FloatTensor(matched_goal_obs).to(device, non_blocking=True)
                ret['matched_goal_obses'].append(matched_goal_obs)

            contact_points = None
            if use_contact:
                contact_points = self.buffer['contact_points'][batch_idxes].squeeze(1)   # B x 3
                obs[:, 0] = contact_points
                contact_points = torch.FloatTensor(contact_points)

            action_idxes = (curr_tool_traj_idx.reshape(-1, 1) * self.horizon + np.arange(self.horizon).reshape(1, -1)).flatten()
            _, pos_tool_trajs = self.get_state(action_idxes)
            pos_tool_trajs = pos_tool_trajs.reshape(len(action_idxes), -1, self.args.dimtool)[:, tid, :]  # use the current tool
            pos_tool_trajs = pos_tool_trajs.reshape(b, self.horizon, -1)
            #### IMPORTANT ####
            pos_tool_trajs[:, 0, 3:7] = INIT_TOOL_POSES[tid]
            #### IMPORTANT ####
            pos_tool_trajs = torch.FloatTensor(pos_tool_trajs)
            if self.args.actor_type == 'v0':
                # deprecated. still use the absolute pose as label. should use relative pose instead 
                pos_tool_trajs = self.tool_traj_to_delta(pos_tool_trajs, contact_points).to(device, non_blocking=True)
            else:
                pos_tool_trajs = pos_tool_trajs.to(device, non_blocking=True)
            success_flag = np.vstack([np.ones((b,1)), np.zeros((b,1))])

            obs = torch.FloatTensor(obs).to(device, non_blocking=True) # B x 1000 x 3
            goal_obs = torch.FloatTensor(goal_obs).to(device, non_blocking=True) # B x 1000 x 3
            if use_contact:
                contact_points = torch.FloatTensor(contact_points).to(device, non_blocking=True) # B x 3
            success_flag = torch.FloatTensor(success_flag).to(device, non_blocking=True) # 2B x 1
            
            ret['obses'].append(obs)
            ret['goal_obses'].append(goal_obs)
            ret['contact_points'].append(contact_points)
            ret['pos_tool_trajs'].append(pos_tool_trajs)
            # ret['neg_tool_trajs'].append(neg_tool_trajs)
            ret['success_flag'].append(success_flag)
        ret['stats'] = self.stats
        return ret

    def sample_transition_twostep(self, batch_tool_idxes, device):
        ret = {}
        for key in ['obses', 'goal_obses', 'pos_tool_trajs', 'start_tool_states', 'mid_tool_states', 'end_tool_states', 'success_flag', 'matched_goal_obses']:
            ret[key] = []
        for tid, batch_idxes in zip(self.args.train_tool_idxes, batch_tool_idxes):
            b = batch_idxes.shape[0]
            init_batch_idxes = self.horizon * (batch_idxes // self.horizon)
            traj_idxes = init_batch_idxes // self.horizon
            
            obs, _ = self.get_state(init_batch_idxes)
            target_v = self.buffer['target_v'][init_batch_idxes]
            goal_obs = self.np_target_pc[target_v]
            if self.args.actor_type == 'tfn_goal_as_flow':
                matched_goal_obs = self.matched_goals[target_v]
                matched_goal_obs = torch.FloatTensor(matched_goal_obs).to(device, non_blocking=True)
                ret['matched_goal_obses'].append(matched_goal_obs)

            action_idxes = (traj_idxes.reshape(-1, 1) * self.horizon + np.arange(self.horizon).reshape(1, -1)).flatten()
            _, pos_tool_trajs = self.get_state(action_idxes)
            pos_tool_trajs = pos_tool_trajs.reshape(len(action_idxes), -1, self.args.dimtool)[:, tid, :]  # use the current tool
            pos_tool_trajs = torch.FloatTensor(pos_tool_trajs.reshape(b, self.horizon, -1)).to(device, non_blocking=True)
            success_flag = np.vstack([np.ones((b,1)), np.zeros((b,1))])

            start_tool_states = self.sample_stacked_tool_states(batch_idxes, self.args.frame_stack, tid)
            mid_idxes = np.array([idx + 1 if idx % self.horizon < self.horizon - 1 else idx for idx in batch_idxes])
            _, mid_tool_states = self.get_state(mid_idxes)
            mid_tool_states = mid_tool_states.reshape(len(mid_idxes), -1, self.args.dimtool)[:, tid, :]  # use the current tool
            end_idxes = np.array([idx + 2 if idx % self.horizon < self.horizon - 2 else idx for idx in batch_idxes])
            _, end_tool_states = self.get_state(end_idxes)
            end_tool_states = end_tool_states.reshape(len(end_idxes), -1, self.args.dimtool)[:, tid, :]  # use the current tool
            start_tool_states = torch.FloatTensor(start_tool_states).to(device, non_blocking=True)
            mid_tool_states = torch.FloatTensor(mid_tool_states).view(b, 1, -1).to(device, non_blocking=True)
            end_tool_states = torch.FloatTensor(end_tool_states).view(b, 1, -1).to(device, non_blocking=True)

            obs = torch.FloatTensor(obs).to(device, non_blocking=True) # B x 1000 x 3
            goal_obs = torch.FloatTensor(goal_obs).to(device, non_blocking=True) # B x 1000 x 3
            success_flag = torch.FloatTensor(success_flag).to(device, non_blocking=True) # 2B x 1
            
            ret['obses'].append(obs)
            ret['goal_obses'].append(goal_obs)
            ret['pos_tool_trajs'].append(pos_tool_trajs)
            ret['start_tool_states'].append(start_tool_states)
            ret['mid_tool_states'].append(mid_tool_states)
            ret['end_tool_states'].append(end_tool_states)
            ret['success_flag'].append(success_flag)
        ret['stats'] = self.stats
        return ret
    
    def sample_stacked_tool_states(self, idx, frame_stack, tid):
        # frame_stack =1 means no stacking
        padded_step = np.concatenate([np.zeros(shape=frame_stack - 1, dtype=np.int), np.arange(self.horizon)])
        traj_idx = idx // self.horizon
        traj_t = idx % self.horizon
        idxes = np.arange(0, frame_stack).reshape(1, -1) + traj_t.reshape(-1, 1)  # TODO For actual stacking, should use negative timestep
        stacked_t = padded_step[idxes]  # B x frame_stack
        stacked_idx = ((traj_idx * self.horizon).reshape(-1, 1) + stacked_t) # B x frame_stack
        stack_obs = self.buffer['states'][stacked_idx, 3000:3000+self.args.num_tools*self.args.dimtool]
        stack_tool_states = stack_obs.reshape(len(idx), frame_stack, -1, self.args.dimtool)[:, :, tid, :] # B x frame_stack x 8
        # stack_obs = np.concatenate(stack_obs, axis=-1)
        return stack_tool_states