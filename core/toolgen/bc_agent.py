import os
import copy
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from chester.simple_logger import LogDict
from chester import logger
from core.diffskill.agents.pointflow_vae import PointFlowVAE
from core.diffskill.agents.pointnet_actor import PointActor, PointActorRecurrent, PointActorConditioned, PointActorTFN, PointActorTFNTraj
from core.diffskill.utils import dict_add_prefix
from core.toolgen.se3 import random_so3
from core.toolgen.visualization.plots import dcp_sg_plot
from core.utils.pc_utils import batch_resample_pc
from core.utils.pcl_visualization_utils import create_pcl_plot, visualize_point_cloud
import glob
from natsort import natsorted
import pytorch3d.transforms as transforms
from core.toolgen.utils import chamfer_distance, bgdR, optimize_reset_pose, optimize_delta_poses
import torch.nn.functional as F
from core.utils.pcl_visualization_utils import create_flow_plot
from tqdm import tqdm


class BCAgent(object):
    def __init__(self, args, device='cuda'):
        # """
        self.args = args
        self.train_stats = {}
        if args.actor_type == 'v0':
            self.actor = PointActor(args, input_dim=2, action_dim=self.args.horizon * 6).to(device)
        elif args.actor_type == 'conditioned':
            self.actor = PointActorConditioned(args, input_dim=3, hidden_dim=64, action_dim=(self.args.horizon-2)*6).to(device)
        elif args.actor_type == 'conditioned_goal_as_flow':
            self.actor = PointActorConditioned(args, input_dim=6, hidden_dim=64, action_dim=(self.args.horizon-2)*6).to(device)
        elif args.actor_type == 'recurrent':
            self.actor = PointActorRecurrent(args, input_dim=3, hidden_dim=64, action_dim=6).to(device)
        elif args.actor_type == 'tfn':
            self.actor = PointActorTFN(args, input_dim=3, action_dim=3).to(device)
        elif args.actor_type == 'tfn_traj_feature':
            self.actor = PointActorTFNTraj(args, input_dim=3, action_dim=3*(self.args.horizon-2)).to(device)
        elif args.actor_type == 'tfn_twostep':
            self.actor = PointActorTFN(args, input_dim=3, action_dim=3).to(device)
        elif args.actor_type == 'tfn_backflow' or args.actor_type == 'tfn_goal_as_flow':
            self.actor = PointActorTFN(args, input_dim=6, action_dim=3).to(device)
        else:
            raise NotImplementedError
        
        if args.value_fn_type == 'separate':
            self.value_fn = PointActor(args, input_dim=2, action_dim=1).to(device)
        elif args.value_fn_type == 'shared':
            self.value_fn = PointActorConditioned(args, input_dim=3, hidden_dim=64, action_dim=1).to(device)
        else:
            raise NotImplementedError
        if args.pointflow_resume_path is not None:
            self.pointflow = PointFlowVAE(args, args.pointflow_resume_path)
        else:
            self.pointflow = None

        self.optim = Adam(list(self.actor.parameters()) +
                          list(self.value_fn.parameters()),
                          lr=args.il_lr)
        # self.tool_particles = [torch.FloatTensor(np.load('plb/envs/tool_pcls/Capsule.npy')).to(device).view(1, 1000, 3), 
        #                         torch.FloatTensor(np.load('plb/envs/tool_pcls/Knife.npy')).to(device).view(1, 1000, 3)]  # TODO CHANGE THIS
        self.tool_pcl_paths = natsorted(glob.glob(os.path.join(os.getcwd(), args.pcl_dir_path, '*.npy')))
        self.tool_particles = [torch.FloatTensor(np.load(path)).to(device).view(1, 1000, 3) for path in self.tool_pcl_paths]
        self.device = device
        self.bgdR = bgdR

    def get_tool_reset_errors(self, data_batch, ret_plot=True, tids=None, reset_poses=None):
        if tids is None:
            tids = self.args.train_tool_idxes
        all_chamfer_dists = []
        all_R_losses = []
        all_t_losses = []
        all_plots = []
        for i, tid in enumerate(tids):
            init_obses, goal_obses, tool_traj, success_flag = data_batch['obses'][i], \
                                                                    data_batch['goal_obses'][i], \
                                                                    data_batch['pos_tool_trajs'][i], \
                                                                    data_batch['success_flag'][i]
            with torch.no_grad():
                B = init_obses.shape[0]
                tool_particles = self.tool_particles[tid].tile(B, 1, 1)
                tool_traj = tool_traj.view(B, self.args.horizon, 6)
                reset_dir = transforms.matrix_to_quaternion(transforms.rotation_6d_to_matrix(tool_traj[:, 0]))
                reset_pos = tool_traj[:, 1, :3]

                data_pc, mean = self.organize_pc_data([init_obses, goal_obses])
                data_tool_pc, _ = self.organize_pc_data_pn(tool_particles)
                if self.args.actor_type == 'v0':
                    pred_traj, _ = self.actor(data_pc, data_tool_pc)
                    pred_traj = pred_traj.view(B, self.args.horizon, 6)
                elif self.args.actor_type == 'conditioned' or self.args.actor_type == 'recurrent':
                        pred_traj, _ = self.actor.init_pose_actor(data_pc, data_tool_pc)
                        pred_traj = pred_traj.view(B, -1, 6)
                else:
                    raise NotImplementedError
                pred_traj[:, 1:2, :] += mean.repeat(1, 1, 2)   # relative to dough com
                pred_traj = pred_traj
                pred_reset_dir = transforms.matrix_to_quaternion(transforms.rotation_6d_to_matrix(pred_traj[:, 0]))
                pred_reset_pos = pred_traj[:, 1, :3]
            
                tool_particles = self.tool_particles[tid]
                pred_tool_particles = transforms.quaternion_apply(pred_reset_dir.unsqueeze(1), tool_particles) + pred_reset_pos.unsqueeze(1)
                gt_tool_particles = transforms.quaternion_apply(reset_dir.unsqueeze(1), tool_particles) + reset_pos.unsqueeze(1)
                dist1, _ = chamfer_distance(pred_tool_particles, gt_tool_particles)
                all_chamfer_dists.append(dist1.cpu().numpy())

                R_pred = transforms.rotation_6d_to_matrix(pred_traj[:, 0])
                R_gt = transforms.rotation_6d_to_matrix(tool_traj[:, 0])
                t_pred = pred_traj[:, 1, :3]
                t_gt = tool_traj[:, 1, :3]
                R_loss = torch.mean(self.bgdR(R_gt, R_pred) * 180/np.pi, dim=0)
                t_loss = torch.mean(torch.norm(t_pred - t_gt, dim=1, p=2), dim=0)
                all_R_losses.append(R_loss.cpu().numpy())
                all_t_losses.append(t_loss.cpu().numpy())
                if ret_plot:
                    for i in range(B):
                        obs = torch.FloatTensor(batch_resample_pc(init_obses[i:i+1].cpu().numpy(), 1300)).cuda()
                        action_clouds = torch.FloatTensor(batch_resample_pc(self.tool_particles[tid].cpu().numpy(), 500)).cuda()
                        anchor_clouds = torch.cat([action_clouds, obs], dim=1)
                        fig = dcp_sg_plot(self.tool_particles[tid][0], anchor_clouds[0], t_gt[i, :3], t_pred[i, :3], R_gt[i], R_pred[i], None)
                        all_plots.append(fig)
        return {
            'chamfer_dists': all_chamfer_dists,
            'R_losses': all_R_losses,
            't_losses': all_t_losses,
            'plots': all_plots
        }
    def rw_act_with_fitting(self, dpc, goal_dpc, tool_pc, save_plots=True):
        # pcl_dir_path = 'plb/envs/tool_pcls/multi_tool'
        # train_tool_paths = natsorted(glob.glob(os.path.join(os.getcwd(), pcl_dir_path, '*.npy')))
        # train_tool_particles = [torch.FloatTensor(np.load(path)).to('cuda').view(1, 1000, 3) for path in train_tool_paths]

        input = torch.cat([dpc, goal_dpc, tool_pc], dim=1)
        curr_tool_pcl = self.pointflow.reconstruct(input, 1000)
        pred_init_pose = torch.empty([1, 2, 6]).cuda()
        pred_init_pose[:, 0, :] = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(torch.FloatTensor([[1, 0, 0, 0]]))).cuda()
        pred_init_pose[:, 1, :] = torch.mean(curr_tool_pcl, dim=1).tile(1, 2)
        init_tool_particles = curr_tool_pcl - torch.mean(curr_tool_pcl, dim=1, keepdim=True)
        visualize_point_cloud(tool_pc.cpu().numpy())

        # fitting the test tool pcl to the predicted tool pcl
        with torch.enable_grad():
            B = 20
            init_pcl = tool_pc.view(1, 1000, 3)
            target_pcl = curr_tool_pcl.tile(B, 1, 1)
            cur_poses  = torch.nn.Parameter(random_so3(B, rot_var=np.pi*2).cuda().view(B, 1, 4), requires_grad=True)
            cur_locs = torch.nn.Parameter(((torch.rand((B, 1, 3)) * 2 - 1)* 0.1).cuda() + pred_init_pose[:, 1:2, :3], requires_grad=True)
            scene_points = torch.cat([dpc, goal_dpc], dim=1)
            opt_reset_pose, opt_tool_pcl, opt_plot = optimize_reset_pose(scene_points, init_pcl, target_pcl, cur_poses, cur_locs)
            with torch.no_grad():
                if 'tfn' not in self.args.actor_type:
                    data_all, _ = self.organize_pc_data_scene([dpc, curr_tool_pcl, goal_dpc])
                    pred_deltas, _ = self.actor(data_all)
                    best_traj = torch.cat([pred_init_pose, pred_deltas], dim=1)
                    best_traj = best_traj.view(self.args.horizon, 6)
                    all_pcls = []
                    cur_dir = transforms.matrix_to_quaternion(transforms.rotation_6d_to_matrix(best_traj[0]))
                    cur_pos = best_traj[1, :3]
                    for i in range(2, len(best_traj)):
                        delta_pos, delta_dir = best_traj[i, :3], best_traj[i, 3:]
                        n_pos = cur_pos + delta_pos
                        n_dir = transforms.quaternion_multiply(
                            transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(delta_dir, "XYZ")), 
                            cur_dir
                            )
                        n_pcl = transforms.quaternion_apply(n_dir.view(1, 1, 4), init_tool_particles) + n_pos.view(1, 1, 3)
                        all_pcls.append(n_pcl[0])
                        cur_pos = n_pos
                        cur_dir = n_dir
                else:
                    assert self.args.actor_type == 'tfn_traj_feature'
                    data_all, _ = self.organize_pc_data_tfn([dpc, curr_tool_pcl, goal_dpc], center=False)
                    pred_traj_flows, pred_pre_svd_flows = self.actor(data_all)
                    all_pcls = []
                    pred_deltas = []
                    for i in range(self.args.horizon-2):
                        pred_flows = pred_traj_flows[:, i, :]
                        R, t = self.actor.get_delta_pose_xyz_flow(curr_tool_pcl[0], pred_pre_svd_flows[0, i, :])
                        if self.actor.scale_pcl_val is not None:
                            curr_tool_pcl = curr_tool_pcl[:, :, :3] + pred_flows / self.actor.scale_pcl_val
                            t = t / self.actor.scale_pcl_val
                        all_pcls.append(curr_tool_pcl[0])
                        pred_deltas.append(torch.cat([t, transforms.matrix_to_euler_angles(R, 'XYZ')], dim=-1))
                    pred_deltas = torch.stack(pred_deltas, dim=1)
                
                with torch.enable_grad():
                    delta_poses = torch.nn.Parameter(torch.zeros((len(all_pcls), 6)).cuda())
                    delta_poses, all_opt_pcls, opt_poses = optimize_delta_poses(all_pcls, init_pcl, delta_poses, opt_reset_pose)
                    visualize_point_cloud([pcl.cpu().numpy() for pcl in all_pcls])
                    visualize_point_cloud(all_opt_pcls.cpu().numpy())
                opt_reset_mat = transforms.quaternion_to_matrix(opt_reset_pose[:, 3:7])
                opt_reset_rot = transforms.matrix_to_rotation_6d(opt_reset_mat).view(1, 1, 6)
                opt_reset_loc = opt_reset_pose[:, :3].tile(1, 2).view(1, 1, 6)
                best_traj = torch.cat([opt_reset_rot, opt_reset_loc, delta_poses.view(1, -1, 6)], dim=1)
                best_traj = best_traj.view(self.args.horizon, 6)
        return torch.vstack([pcl.view(1, 1000, 3).cpu() for pcl in all_pcls]), all_opt_pcls.cpu(), opt_plot

    def act_with_fitting(self, state, goal_dpc, allowed_tids, save_plots=True):
        # pcl_dir_path = 'plb/envs/tool_pcls/multi_tool'
        # train_tool_paths = natsorted(glob.glob(os.path.join(os.getcwd(), pcl_dir_path, '*.npy')))
        # train_tool_particles = [torch.FloatTensor(np.load(path)).to('cuda').view(1, 1000, 3) for path in train_tool_paths]
        dpc = state[:, :3000].reshape(-1, 1000, 3)

        val_dict = {}
        data_pc, dpc_mean = self.organize_pc_data([dpc, goal_dpc])
        # tids = range(len(self.tool_particles))
        for tid in allowed_tids:
            cur_tool_particles = self.tool_particles[tid]
            tool_state_idx = 3000+tid*self.args.dimtool
            tool_state = state[:, tool_state_idx:tool_state_idx+8].reshape(1, 1, 8)
            cur_tool_particles = transforms.quaternion_apply(tool_state[:, :, 3:7], cur_tool_particles)
            cur_data_tool_pc, _ = self.organize_pc_data_pn(cur_tool_particles)
            assert self.args.value_fn_type == 'separate'
            curr_val, _ = self.value_fn(data_pc, cur_data_tool_pc)
            curr_val = curr_val.item()
            print(f"tid: {tid}, val: {curr_val}")
            val_dict[tid] = curr_val
        
        best_tid = max(val_dict, key=val_dict.get)
        tool_pc = self.tool_particles[best_tid].view(1, 1000, 3)
        tool_pc = tool_pc - torch.mean(tool_pc, dim=0, keepdim=True)
        input = torch.cat([dpc, goal_dpc, tool_pc], dim=1)
        curr_tool_pcl = self.pointflow.reconstruct(input, 1000)
        pred_init_pose = torch.empty([1, 2, 6]).cuda()
        pred_init_pose[:, 0, :] = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(torch.FloatTensor([[1, 0, 0, 0]]))).cuda()
        pred_init_pose[:, 1, :] = torch.mean(curr_tool_pcl, dim=1).tile(1, 2)
        init_tool_particles = curr_tool_pcl - torch.mean(curr_tool_pcl, dim=1, keepdim=True)

        # fitting the test tool pcl to the predicted tool pcl
        with torch.enable_grad():
            B = 20
            init_pcl = self.tool_particles[best_tid].view(1, 1000, 3)
            tool_state_idx = 3000+best_tid*self.args.dimtool
            tool_state = state[:, tool_state_idx:tool_state_idx+8].reshape(1, 1, 8)
            init_pcl = transforms.quaternion_apply(tool_state[:, :, 3:7], init_pcl)
            target_pcl = curr_tool_pcl.tile(B, 1, 1)
            cur_poses  = torch.nn.Parameter(random_so3(B, rot_var=np.pi*2).cuda().view(B, 1, 4), requires_grad=True)
            cur_locs = torch.nn.Parameter(((torch.rand((B, 1, 3)) * 2 - 1)* 0.1).cuda() + pred_init_pose[:, 1:2, :3], requires_grad=True)
            scene_points = torch.cat([dpc, goal_dpc], dim=1)
            opt_reset_pose, opt_tool_pcl, opt_plot = optimize_reset_pose(scene_points, init_pcl, target_pcl, cur_poses, cur_locs)
        with torch.no_grad():
            if self.args.opt_traj_deltas:
                if 'tfn' not in self.args.actor_type:
                    data_all, _ = self.organize_pc_data_scene([dpc, curr_tool_pcl, goal_dpc])
                    pred_deltas, _ = self.actor(data_all)
                    best_traj = torch.cat([pred_init_pose, pred_deltas], dim=1)
                    best_traj = best_traj.view(self.args.horizon, 6)
                    all_pcls = []
                    cur_dir = transforms.matrix_to_quaternion(transforms.rotation_6d_to_matrix(best_traj[0]))
                    cur_pos = best_traj[1, :3]
                    for i in range(2, len(best_traj)):
                        delta_pos, delta_dir = best_traj[i, :3], best_traj[i, 3:]
                        n_pos = cur_pos + delta_pos
                        n_dir = transforms.quaternion_multiply(
                            transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(delta_dir, "XYZ")), 
                            cur_dir
                            )
                        n_pcl = transforms.quaternion_apply(n_dir.view(1, 1, 4), init_tool_particles) + n_pos.view(1, 1, 3)
                        all_pcls.append(n_pcl[0])
                        cur_pos = n_pos
                        cur_dir = n_dir
                else:
                    assert self.args.actor_type == 'tfn_traj_feature'
                    data_all, _ = self.organize_pc_data_tfn([dpc, curr_tool_pcl, goal_dpc], center=False)
                    pred_traj_flows, pred_pre_svd_flows = self.actor(data_all)
                    all_pcls = []
                    pred_deltas = []
                    for i in range(self.args.horizon-2):
                        pred_flows = pred_traj_flows[:, i, :]
                        R, t = self.actor.get_delta_pose_xyz_flow(curr_tool_pcl[0], pred_pre_svd_flows[0, i, :])
                        if self.actor.scale_pcl_val is not None:
                            curr_tool_pcl = curr_tool_pcl[:, :, :3] + pred_flows / self.actor.scale_pcl_val
                            t = t / self.actor.scale_pcl_val
                        all_pcls.append(curr_tool_pcl[0])
                        pred_deltas.append(torch.cat([t, transforms.matrix_to_euler_angles(R, 'XYZ')], dim=-1))
                    pred_deltas = torch.stack(pred_deltas, dim=1)

               
                with torch.enable_grad():
                    delta_poses = torch.nn.Parameter(torch.zeros((len(all_pcls), 6)).cuda())
                    delta_poses, all_opt_pcls, opt_poses = optimize_delta_poses(all_pcls, init_pcl, delta_poses, opt_reset_pose)
                    # visualize_point_cloud([pcl.cpu().numpy() for pcl in all_pcls])
                    # visualize_point_cloud(all_opt_pcls.cpu().numpy())
                ### added: important???
                opt_reset_pose[:, 3:7] = transforms.quaternion_multiply(opt_reset_pose[:, 3:7], tool_state[0, :, 3:7])
                ###
                opt_reset_mat = transforms.quaternion_to_matrix(opt_reset_pose[:, 3:7])
                opt_reset_rot = transforms.matrix_to_rotation_6d(opt_reset_mat).view(1, 1, 6)
                opt_reset_loc = opt_reset_pose[:, :3].tile(1, 2).view(1, 1, 6)
                best_traj = torch.cat([opt_reset_rot, opt_reset_loc, delta_poses.view(1, -1, 6)], dim=1)
                best_traj = best_traj.view(self.args.horizon, 6)
            else:
                if 'tfn' not in self.args.actor_type:
                    data_all, _ = self.organize_pc_data_scene([dpc, opt_tool_pcl, goal_dpc])
                    pred_deltas, _ = self.actor(data_all)
                    
                else:
                    data_all, _ = self.organize_pc_data_tfn([dpc, opt_tool_pcl, goal_dpc], center=False)
                    pred_traj_flows, pred_pre_svd_flows = self.actor(data_all)
                    all_pcls = []
                    pred_deltas = []
                    for i in range(self.args.horizon-2):
                        pred_flows = pred_traj_flows[:, i, :]
                        R, t = self.actor.get_delta_pose_xyz_flow(curr_tool_pcl[0], pred_pre_svd_flows[0, i, :])
                        if self.actor.scale_pcl_val is not None:
                            curr_tool_pcl = curr_tool_pcl[:, :, :3] + pred_flows / self.actor.scale_pcl_val
                            t = t / self.actor.scale_pcl_val
                        all_pcls.append(curr_tool_pcl[0])
                        pred_deltas.append(torch.cat([t, transforms.matrix_to_euler_angles(R, 'XYZ')], dim=-1))
                    pred_deltas = torch.stack(pred_deltas, dim=1)
                ### added: important???
                opt_reset_pose[:, 3:7] = transforms.quaternion_multiply(opt_reset_pose[:, 3:7], tool_state[0, :, 3:7])
                ###
                opt_reset_mat = transforms.quaternion_to_matrix(opt_reset_pose[:, 3:7])
                opt_reset_rot = transforms.matrix_to_rotation_6d(opt_reset_mat).view(1, 1, 6)
                opt_reset_loc = opt_reset_pose[:, :3].tile(1, 2).view(1, 1, 6)
                best_traj = torch.cat([opt_reset_rot, opt_reset_loc, pred_deltas], dim=1)
                best_traj = best_traj.view(self.args.horizon, 6)
        if save_plots:
            return best_traj, best_tid, opt_plot
        else:
            assert False

    def act(self, state, goal_dpc, allowed_tids, save_plots=False):
        dpc = state[:, :3000].reshape(-1, 1000, 3)
        plots = []
        val_dict = {}
        data_pc, dpc_mean = self.organize_pc_data([dpc, goal_dpc])
        # allowed_tids = [0]
        for tid in allowed_tids:
            cur_tool_particles = self.tool_particles[tid]
            tool_state_idx = 3000+tid*self.args.dimtool
            tool_state = state[:, tool_state_idx:tool_state_idx+8].reshape(1, 1, 8)
            cur_tool_particles = transforms.quaternion_apply(tool_state[:, :, 3:7], cur_tool_particles)
            cur_data_tool_pc, _ = self.organize_pc_data_pn(cur_tool_particles)
            if self.args.value_fn_type == 'separate':
                curr_val, _ = self.value_fn(data_pc, cur_data_tool_pc)
            curr_val = curr_val.item()
            print(f"tid: {tid}, val: {curr_val}")
            val_dict[tid] = curr_val
        
        best_tid = max(val_dict, key=val_dict.get)
        tool_particles = self.tool_particles[best_tid]
        tool_state_idx = 3000+best_tid*self.args.dimtool
        tool_state = state[:, tool_state_idx:tool_state_idx+8].reshape(1, 1, 8)
        tool_particles = transforms.quaternion_apply(tool_state[:, :, 3:7], tool_particles)
        data_tool_pc, _ = self.organize_pc_data_pn(tool_particles)

        if self.args.actor_type == 'v0':
            best_traj, _ = self.actor(data_pc, data_tool_pc)
            best_traj.view(1, self.args.horizon, 6)[:, 1:2, :] += dpc_mean.repeat(1, 1, 2)  # relative to dough com -> relative to world
            best_traj = best_traj.view(self.args.horizon, 6)
            ### IMPORTANT FOR OUTPUTTING RELATIVE RESET POSES ###
            if self.args.random_init_pose:
                pred_reset_pose = transforms.quaternion_multiply(
                                transforms.matrix_to_quaternion(
                                transforms.rotation_6d_to_matrix(best_traj[0:1, :].clone())
                                ),
                                tool_state[0, :, 3:7])  # relative pose -> absolute pose
                best_traj[0:1, :] = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(pred_reset_pose)).view(1, 6)
            ### IMPORTANT FOR OUTPUTTING RELATIVE RESET POSES ###
        elif self.args.actor_type == 'conditioned' or self.args.actor_type == 'recurrent' or self.args.actor_type == 'conditioned_goal_as_flow':
            pred_init_pose, _ = self.actor.init_pose_actor(data_pc, data_tool_pc)
            pred_init_pose = pred_init_pose.view(1, -1, 6)
            ### IMPORTANT FOR OUTPUTTING RELATIVE RESET POSES ###
            if self.args.random_init_pose:
                pred_reset_pose = transforms.quaternion_multiply(
                                transforms.matrix_to_quaternion(
                                transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                                ),
                                tool_state[:, :, 3:7])  # relative pose -> absolute pose
                pred_init_pose[:, 0:1, :] = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(pred_reset_pose)).view(1, 1, 6)
                curr_tool_pcl = transforms.quaternion_apply(pred_reset_pose, tool_particles) # B x 1 x 4 apply to B x 1000 x 3
            ### IMPORTANT FOR OUTPUTTING RELATIVE RESET POSES ###

            curr_tool_pcl = transforms.quaternion_apply(
                            transforms.matrix_to_quaternion(
                            transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                            ), tool_particles) # B x 1 x 4 apply to B x 1000 x 3
            pred_init_pose[:, 1:2, :] += dpc_mean.repeat(1, 1, 2)   # relative to dough com -> relative to world
            curr_tool_pcl = curr_tool_pcl + pred_init_pose[:, 1:2, :3].clone()
            if self.args.actor_type == 'conditioned_goal_as_flow':
                curr_tool_pcl = torch.cat([curr_tool_pcl, torch.zeros_like(curr_tool_pcl).cuda()], dim=-1)
                goal_flow = goal_dpc - dpc
                dpc = torch.cat([dpc, goal_flow], dim=-1)
                data_all, _ = self.organize_pc_data_tfn_goalflow([dpc, curr_tool_pcl])
            else:
                data_all, _ = self.organize_pc_data_scene([dpc, curr_tool_pcl, goal_dpc])
            if self.args.actor_type == 'conditioned' or self.args.actor_type == 'conditioned_goal_as_flow':
                pred_deltas, _ = self.actor(data_all)
            else:
                pred_deltas, _ = self.actor(data_all, horizon=self.args.horizon-2)
            best_traj = torch.cat([pred_init_pose, pred_deltas], dim=1)
        elif 'tfn' in self.args.actor_type:
            pred_init_pose, _ = self.actor.init_pose_actor(data_pc, data_tool_pc)
            pred_init_pose = pred_init_pose.view(1, -1, 6)
            ### IMPORTANT FOR OUTPUTTING RELATIVE RESET POSES ###
            if self.args.random_init_pose:
                pred_reset_pose = transforms.quaternion_multiply(
                                transforms.matrix_to_quaternion(
                                transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                                ),
                                tool_state[:, :, 3:7])  # relative pose -> absolute pose
                pred_init_pose[:, 0:1, :] = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(pred_reset_pose)).view(1, 1, 6)
                curr_tool_pcl = transforms.quaternion_apply(pred_reset_pose, tool_particles) # B x 1 x 4 apply to B x 1000 x 3
            ### IMPORTANT FOR OUTPUTTING RELATIVE RESET POSES ###

            curr_tool_pcl = transforms.quaternion_apply(
                            transforms.matrix_to_quaternion(
                            transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                            ), tool_particles) # B x 1 x 4 apply to B x 1000 x 3
            pred_init_pose[:, 1:2, :] += dpc_mean.repeat(1, 1, 2)   # relative to dough com -> relative to world
            curr_tool_pcl = curr_tool_pcl + pred_init_pose[:, 1:2, :3].clone()
            if self.args.actor_type == 'tfn_backflow':
                curr_tool_pcl = torch.cat([curr_tool_pcl, torch.zeros_like(curr_tool_pcl).cuda()], dim=-1)
                input_dpc = torch.cat([dpc, torch.zeros_like(dpc).cuda()], dim=-1)
                input_goal_dpc = torch.cat([goal_dpc, torch.zeros_like(goal_dpc).cuda()], dim=-1)
            elif self.args.actor_type == 'tfn_goal_as_flow':
                goal_flow = goal_dpc - dpc
                input_dpc = torch.cat([dpc, goal_flow], dim=-1)
                curr_tool_pcl = torch.cat([curr_tool_pcl, torch.zeros_like(curr_tool_pcl).cuda()], dim=-1)
            else:
                input_dpc = dpc
                input_goal_dpc = goal_dpc
            pred_deltas = []
            pred_traj = self.args.actor_type == 'tfn_traj_feature'
            if pred_traj:
                data_all, _ = self.organize_pc_data_tfn([input_dpc, curr_tool_pcl, input_goal_dpc], center=False)
                pred_traj_flows, pred_pre_svd_flows = self.actor(data_all)
            for i in range(self.args.horizon-2):
                if not pred_traj:
                    if self.args.actor_type == 'tfn_goal_as_flow':
                        data_all, _ = self.organize_pc_data_tfn_goalflow([input_dpc, curr_tool_pcl], center=False)
                    else:
                        data_all, _ = self.organize_pc_data_tfn([input_dpc, curr_tool_pcl, input_goal_dpc], center=False)
                    pred_flows, _ = self.actor(data_all)
                    R, t = self.actor.get_delta_pose(data_all)
                else:
                    pred_flows = pred_traj_flows[:, i, :]
                    R, t = self.actor.get_delta_pose_xyz_flow(curr_tool_pcl[0], pred_pre_svd_flows[0, i, :])
                if save_plots:
                    plot_points = torch.cat([curr_tool_pcl[:, :, :3], input_dpc[:,:,:3], goal_dpc], dim=1)
                    fig = create_flow_plot(plot_points[0].detach().cpu().numpy(), pred_flows[0].detach().cpu().numpy(), flow_scale=50.)
                    plots.append(fig)
                if self.actor.scale_pcl_val is not None:
                    curr_tool_pcl = curr_tool_pcl[:, :, :3] + pred_flows / self.actor.scale_pcl_val
                    t = t / self.actor.scale_pcl_val
                pred_deltas.append(torch.cat([t, transforms.matrix_to_euler_angles(R, 'XYZ')], dim=-1))
                if self.args.actor_type == 'tfn_backflow':
                    tool_back_flow = -pred_flows
                    curr_tool_pcl = torch.cat([curr_tool_pcl[:, :, :3], tool_back_flow], dim=-1)
                if self.args.actor_type == 'tfn_goal_as_flow':
                    curr_tool_pcl = torch.cat([curr_tool_pcl[:, :, :3], torch.zeros((1, 1000, 3)).cuda()], dim=-1)
            best_traj = torch.cat([pred_init_pose, torch.stack(pred_deltas, dim=1)], dim=1)
        else:
            raise NotImplementedError
        if save_plots:
            return best_traj.view(self.args.horizon, 6), best_tid, plots
        else:
            return best_traj.view(self.args.horizon, 6), best_tid
    
    def organize_pc_data(self, pcs, center=True, std=None):
        dpc = pcs[0]
        obs = torch.cat(pcs, dim=1)  # concat the batch for point cloud
        # Preprocess obs into the shape that encoder requires
        b, n, f = obs.shape[0], obs.shape[1], obs.shape[2]
        if center:
            mean = torch.mean(dpc, dim=1, keepdim=True)
        else:
            mean = torch.zeros((b, 1, 3)).cuda()
        obs = obs - mean
        scale, _ = torch.max(torch.norm(obs, dim=-1, keepdim=True), dim=1, keepdim=True)
        obs = obs / scale
        # if std is not None:
            # obs = obs / std
        data = {}
        pos, feature = torch.split(obs, [3, f - 3], dim=-1)
        # dough: [0,1] target: 1,0
        onehot = torch.zeros((b, n, 2)).to(obs.device)
        onehot[:, :1000, 1:2] += 1
        onehot[:, 1000:, 0:1] += 1
        x = torch.cat([feature, onehot], dim=-1)
        data['x'] = x.reshape(-1, 2 + feature.shape[-1])
        data['pos'] = pos.reshape(-1, 3)
        data['batch'] = torch.arange(b).repeat_interleave(n).to(obs.device, non_blocking=True)
        return data, mean
    
    def organize_pc_data_scene(self, pcs, center=True):
        dpc = pcs[0]
        obs = torch.cat(pcs, dim=1)  # concat the batch for point cloud
        # Preprocess obs into the shape that encoder requires
        b, n, f = obs.shape[0], obs.shape[1], obs.shape[2]
        if center:
            mean = torch.mean(dpc, dim=1, keepdim=True)
        else:
            mean = torch.zeros((b, 1, 3)).cuda()
        obs = obs - mean
        scale, _ = torch.max(torch.norm(obs, dim=-1, keepdim=True), dim=1, keepdim=True)
        obs = obs / scale
        # visualize_point_cloud([obs[0, :1000].detach().cpu().numpy(), obs[0, 1000:].detach().cpu().numpy()])
        data = {}
        pos, feature = torch.split(obs, [3, f - 3], dim=-1)
        # dough: [0,0,1] tool: 0,1,0, target: 1,0,0
        n = obs.shape[0]
        onehot = torch.zeros((obs.shape[0], obs.shape[1], 3)).to(obs.device)
        onehot[:, :1000, 2:3] += 1
        onehot[:, 1000:2000, 1:2] += 1
        onehot[:, 2000:, 0:1] += 1
        x = torch.cat([feature, onehot], dim=-1)
        data['x'] = x.reshape(-1, 3 + feature.shape[-1])
        data['pos'] = pos.reshape(-1, 3)
        data['batch'] = torch.arange(n).repeat_interleave(obs.shape[1]).to(obs.device, non_blocking=True)
        return data, mean
    
    def organize_pc_data_tfn(self, pcs, center=True):
        dpc = pcs[0]
        obs = torch.cat(pcs, dim=1)  # concat the batch for point cloud
        # Preprocess obs into the shape that encoder requires
        b, n, f = obs.shape[0], obs.shape[1], obs.shape[2]
        pos, feature = torch.split(obs, [3, f - 3], dim=-1)
        if center:
            mean = torch.mean(dpc[:, :, :3], dim=1, keepdim=True)
            scale, _ = torch.max(torch.norm(pos, dim=-1, keepdim=True), dim=1, keepdim=True)
        else:
            mean = torch.zeros((b, 1, 3)).cuda()
            scale = 1.
        pos = pos - mean
        pos = pos / scale
        # visualize_point_cloud([obs[0, :1000].detach().cpu().numpy(), obs[0, 1000:].detach().cpu().numpy()])
        data = {}
        # dough: [0,0,1] tool: 0,1,0, target: 1,0,0
        n = obs.shape[0]
        onehot = torch.zeros((obs.shape[0], obs.shape[1], 3)).to(obs.device)
        onehot[:, :1000, 2:3] += 1
        onehot[:, 1000:2000, 1:2] += 1
        onehot[:, 2000:, 0:1] += 1
        x = torch.cat([feature, onehot], dim=-1)
        data['x'] = x.reshape(-1, 3 + feature.shape[-1])
        data['pos'] = pos.reshape(-1, 3)
        data['batch'] = torch.arange(n).repeat_interleave(obs.shape[1]).to(obs.device, non_blocking=True)
        
        ptrs = []
        ptr = 0
        for ob in obs:
            ptrs.append(ptr)
            ptr += ob.shape[0]
        ptrs.append(ptr)
        data['ptr'] = torch.tensor(ptrs).to(torch.int32).to(obs.device, non_blocking=True)
        return data, mean
    
    def organize_pc_data_tfn_goalflow(self, pcs, center=True):
        dpc = pcs[0]
        obs = torch.cat(pcs, dim=1)  # concat the batch for point cloud
        # Preprocess obs into the shape that encoder requires
        b, n, f = obs.shape[0], obs.shape[1], obs.shape[2]
        pos, feature = torch.split(obs, [3, f - 3], dim=-1)
        if center:
            mean = torch.mean(dpc[:, :, :3], dim=1, keepdim=True)
            scale, _ = torch.max(torch.norm(pos, dim=-1, keepdim=True), dim=1, keepdim=True)
        else:
            mean = torch.zeros((b, 1, 3)).cuda()
            scale = 1.
        pos = pos - mean
        pos = pos / scale
        # visualize_point_cloud([obs[0, :1000].detach().cpu().numpy(), obs[0, 1000:].detach().cpu().numpy()])
        data = {}
        # dough + flow: [0,0,1] tool: [0,1,0]
        n = obs.shape[0]
        onehot = torch.zeros((obs.shape[0], obs.shape[1], 3)).to(obs.device)
        onehot[:, :1000, 2:3] += 1
        onehot[:, 1000:, 1:2] += 1
        
        x = torch.cat([feature, onehot], dim=-1)
        data['x'] = x.reshape(-1, 3 + feature.shape[-1])
        data['pos'] = pos.reshape(-1, 3)
        data['batch'] = torch.arange(n).repeat_interleave(obs.shape[1]).to(obs.device, non_blocking=True)
        
        ptrs = []
        ptr = 0
        for ob in obs:
            ptrs.append(ptr)
            ptr += ob.shape[0]
        ptrs.append(ptr)
        data['ptr'] = torch.tensor(ptrs).to(torch.int32).to(obs.device, non_blocking=True)
        return data, mean


    def update_train_stats(self, stats):
        self.train_stats = stats

    def organize_pc_data_pn(self, obs, center=True, std=None):
        # Preprocess obs into the shape that encoder requires
        b, n, f = obs.shape[0], obs.shape[1], obs.shape[2]
        if center:
            mean = torch.mean(obs, dim=1, keepdim=True)
        else:
            mean = torch.zeros((b, 1, 3)).cuda()
        obs = obs - mean
        scale, _ = torch.max(torch.norm(obs, dim=-1, keepdim=True), dim=1, keepdim=True)
        obs = obs / scale
        data = {}
        pos, feature = torch.split(obs, [3, f - 3], dim=-1)
        data['x'] = None
        data['pos'] = pos.reshape(-1, 3)
        data['batch'] = torch.arange(b).repeat_interleave(n).to(obs.device, non_blocking=True)
        return data, mean

    def train_pc(self, data_batch, mode='train', epoch=None, ret_plot=False):
        log_dict = LogDict()
        tids = self.args.train_tool_idxes
        noise = mode == 'train'
        for i, tid in enumerate(tids):
            tool_log_dict = LogDict()
            init_obses, goal_obses, tool_traj, success_flag = data_batch['obses'][i], \
                                                                    data_batch['goal_obses'][i], \
                                                                    data_batch['pos_tool_trajs'][i], \
                                                                    data_batch['success_flag'][i]
            B = init_obses.shape[0]
            tool_particles = self.tool_particles[tid].tile(B, 1, 1)
            data_pc, mean = self.organize_pc_data([init_obses, goal_obses])
            data_tool_pc, _ = self.organize_pc_data_pn(tool_particles)

            # if self.args.random_init_pose:
            #     # generate random rotations from so(3)
            #     random_quats = random_so3(B, rot_var=np.pi*2).view(B, 1, 4).cuda()
            #     tool_particles = transforms.quaternion_apply(random_quats, tool_particles)
            #     tool_traj[:, 0:1, 3:7] = transforms.quaternion_apply(random_quats, tool_traj[:, 0:1, 3:7])
            if 'policy' in self.args.train_modules:
                if self.args.actor_type == "v0":
                    pred_traj, _ = self.actor(data_pc, data_tool_pc)
                    tool_traj.view(B, self.args.horizon, 6)[:, 1:2, :] -= mean.repeat(1, 1, 2)   # relative to dough com
                    action_losses = self.actor.traj_action_loss(pred_traj, tool_traj)
                elif "conditioned" in self.args.actor_type:
                    tool_traj.view(B, self.args.horizon, self.args.dimtool)[:, :, :3] -= mean   # relative to dough com
                    pred_init_pose, _ = self.actor.init_pose_actor(data_pc, data_tool_pc)
                    ### IMPORTANT ###
                    # pred_reset_pose = transforms.quaternion_multiply(
                    #                                 transforms.matrix_to_quaternion(
                    #                                 transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                    #                             ), 
                    #                             tool_traj[:, 0:1, 3:7])
                    # moved_tool_particles = transforms.quaternion_apply(pred_reset_pose, tool_particles) # B x 1 x 4 apply to B x 1000 x 3
                    # pred_init_pose[:, 0:1, :] = transforms.quaternion_to_matrix(pred_reset_pose)
                    ### IMPORTANT ###=
                    pred_init_pose = pred_init_pose.view(B, -1, 6)
                    moved_tool_particles = transforms.quaternion_apply(
                                            transforms.matrix_to_quaternion(
                                            transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                                            ), tool_particles) # B x 1 x 4 apply to B x 200 x 3
                    moved_tool_particles = moved_tool_particles + pred_init_pose[:, 1:2, :3].clone() + mean
                    if self.args.actor_type == 'conditioned_goal_as_flow':
                        moved_tool_particles = torch.cat([moved_tool_particles, torch.zeros_like(moved_tool_particles).cuda()], dim=-1)
                        matched_goals = data_batch['matched_goal_obses'][i]
                        goal_flow = matched_goals - init_obses
                        init_obses = torch.cat([init_obses, goal_flow], dim=-1)
                        data_all, _ = self.organize_pc_data_tfn_goalflow([init_obses, moved_tool_particles])
                    else:
                        data_all, _ = self.organize_pc_data_scene([init_obses, moved_tool_particles, goal_obses])
                    pred_deltas, _ = self.actor(data_all)
                    action_losses = self.actor.traj_action_loss(pred_init_pose, pred_deltas, tool_traj)
                elif self.args.actor_type == "recurrent":
                    tool_traj.view(B, self.args.horizon, self.args.dimtool)[:, :, :3] -= mean   # relative to dough com
                    pred_init_pose, _ = self.actor.init_pose_actor(data_pc, data_tool_pc)
                    pred_init_pose = pred_init_pose.view(B, -1, 6)
                    moved_tool_particles = transforms.quaternion_apply(
                                            transforms.matrix_to_quaternion(
                                            transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                                            ), tool_particles) # B x 1 x 4 apply to B x 200 x 3
                    moved_tool_particles = moved_tool_particles + pred_init_pose[:, 1:2, :3].clone() + mean
                    data_all, _ = self.organize_pc_data_scene([init_obses, moved_tool_particles, goal_obses])
                    _, pred_traj = self.actor(data_all, horizon=self.args.horizon-2)
                    action_losses = self.actor.traj_action_loss(pred_init_pose, pred_traj, tool_traj)
                elif 'tfn' in self.args.actor_type:
                    tool_traj.view(B, self.args.horizon, self.args.dimtool)[:, :, :3] -= mean   # relative to dough com
                    pred_init_pose, _ = self.actor.init_pose_actor(data_pc, data_tool_pc)
                    pred_init_pose = pred_init_pose.view(B, -1, 6)
                    moved_tool_particles = transforms.quaternion_apply(
                                            transforms.matrix_to_quaternion(
                                            transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                                            ), tool_particles) # B x 1 x 4 apply to B x 200 x 3
                    moved_tool_particles = moved_tool_particles + pred_init_pose[:, 1:2, :3].clone() + mean
                    action_losses = self.actor.reset_action_loss(pred_init_pose, tool_traj)
                    tool_traj.view(B, self.args.horizon, self.args.dimtool)[:, :, :3] += mean   # relative to dough com
                    
                    if self.args.actor_type == 'tfn':
                        flow_loss = 0
                        curr_tool_pcl = moved_tool_particles.detach()
                        for t in range(2, self.args.horizon):
                            tool_pcl_n = transforms.quaternion_apply(tool_traj[:, t:t+1, 3:7], tool_particles) + tool_traj[:, t:t+1, :3]
                            if self.args.wrt_pred:
                                # data as demo
                                actual_flows = tool_pcl_n - curr_tool_pcl
                                data_all, _ = self.organize_pc_data_tfn([init_obses, curr_tool_pcl, goal_obses], center=False)
                                pred_flows, per_pt_flows = self.actor(data_all)
                                curr_tool_pcl = curr_tool_pcl + pred_flows.detach() / self.actor.scale_pcl_val
                            else:
                                # fully teacher forcing
                                tool_pcl = transforms.quaternion_apply(tool_traj[:, t-1:t, 3:7], tool_particles) + tool_traj[:, t-1:t, :3]
                                actual_flows = tool_pcl_n - tool_pcl
                                data_all, _ = self.organize_pc_data_tfn([init_obses, tool_pcl, goal_obses], center=False)
                                pred_flows, per_pt_flows = self.actor(data_all)

                            if self.actor.scale_pcl_val is not None:    
                                actual_flows = actual_flows * self.actor.scale_pcl_val

                            print('flow_magnitudes', torch.norm(actual_flows, dim=-1).mean().item(), torch.norm(pred_flows, dim=-1).mean().item())
                            
                            # visualize_point_cloud([tool_pcl_n[0].detach().cpu().numpy(), moved_tool_particles[0].detach().cpu().numpy()])
                            flow_loss += self.actor.tfn_action_loss(pred_flows, per_pt_flows, actual_flows)
                    elif self.args.actor_type == 'tfn_traj_feature':
                        flow_loss = 0
                        curr_tool_pcl = moved_tool_particles.detach()
                        data_all, _ = self.organize_pc_data_tfn([init_obses, curr_tool_pcl, goal_obses], center=False)
                        pred_flows, per_pt_flows = self.actor(data_all)
                        for t in range(self.args.horizon-2):
                            tool_pcl_n = transforms.quaternion_apply(tool_traj[:, t:t+1, 3:7], tool_particles) + tool_traj[:, t:t+1, :3]
                            actual_flows = tool_pcl_n - curr_tool_pcl
                            curr_tool_pcl = curr_tool_pcl + pred_flows[:, t, :].detach() / self.actor.scale_pcl_val
                            if self.actor.scale_pcl_val is not None:    
                                actual_flows = actual_flows * self.actor.scale_pcl_val
                            flow_loss += self.actor.tfn_action_loss(pred_flows[:, t, :], per_pt_flows[:, t, :], actual_flows)

                    elif self.args.actor_type == 'tfn_twostep':
                        start_tool_states, end_tool_states = data_batch['start_tool_states'][i], data_batch['end_tool_states'][i]
                        tool_pcl_n = transforms.quaternion_apply(end_tool_states[:, :, 3:7], tool_particles) + end_tool_states[:, :, :3]
                        tool_pcl = transforms.quaternion_apply(start_tool_states[:, :, 3:7], tool_particles) + start_tool_states[:, :, :3]
                        with torch.no_grad():
                            data_all, _ = self.organize_pc_data_tfn([init_obses, tool_pcl, goal_obses], center=False)
                            pred_flows_nograd, _ = self.actor(data_all)
                            pred_tool_pcl = tool_pcl + pred_flows_nograd / self.actor.scale_pcl_val
                            actual_flows = tool_pcl_n - pred_tool_pcl
                        data_all, _ = self.organize_pc_data_tfn([init_obses, pred_tool_pcl, goal_obses], center=False)
                        pred_flows, per_pt_flows = self.actor(data_all)
                        
                        if self.actor.scale_pcl_val is not None:    
                            actual_flows = actual_flows * self.actor.scale_pcl_val
                        print('flow_magnitudes', torch.norm(actual_flows, dim=-1).mean().item(), torch.norm(pred_flows, dim=-1).mean().item())
                        flow_loss = self.actor.tfn_action_loss(pred_flows, per_pt_flows, actual_flows)
                    elif self.args.actor_type == 'tfn_backflow':
                        # BUG HERE
                        assert self.args.frame_stack == 2
                        start_tool_states, end_tool_states = data_batch['start_tool_states'][i], data_batch['end_tool_states'][i]
                        tool_pcl_n = transforms.quaternion_apply(end_tool_states[:, :, 3:7], tool_particles) + end_tool_states[:, :, :3]
                        tool_pcl = transforms.quaternion_apply(start_tool_states[:, -1:, 3:7], tool_particles) + start_tool_states[:, -1:, :3]
                        tool_pcl_hist = transforms.quaternion_apply(start_tool_states[:, 0:1, 3:7], tool_particles) + start_tool_states[:, 0:1, :3]
                        tool_pcl_backward = tool_pcl_hist - tool_pcl
                        if self.actor.scale_pcl_val is not None:
                            tool_pcl_backward = tool_pcl_backward * self.actor.scale_pcl_val
                        tool_pcl = torch.cat([tool_pcl, tool_pcl_backward], dim=-1)
                        init_obses = torch.cat([init_obses, torch.zeros_like(init_obses).cuda()], dim=-1)
                        goal_obses = torch.cat([goal_obses, torch.zeros_like(goal_obses).cuda()], dim=-1)
                        with torch.no_grad():
                            data_all, _ = self.organize_pc_data_tfn([init_obses, tool_pcl, goal_obses], center=False)
                            pred_flows_nograd, _ = self.actor(data_all)
                            pred_tool_pcl = torch.cat([tool_pcl[:, :, :3] + pred_flows_nograd / self.actor.scale_pcl_val, -pred_flows_nograd], dim=-1)
                            actual_flows = tool_pcl_n - pred_tool_pcl[:, :, :3]
                        data_all, _ = self.organize_pc_data_tfn([init_obses, pred_tool_pcl, goal_obses], center=False)
                        pred_flows, per_pt_flows = self.actor(data_all)
                        if self.actor.scale_pcl_val is not None:    
                            actual_flows = actual_flows * self.actor.scale_pcl_val
                        print('flow_magnitudes', torch.norm(actual_flows, dim=-1).mean().item(), torch.norm(pred_flows, dim=-1).mean().item())
                        flow_loss = self.actor.tfn_action_loss(pred_flows, per_pt_flows, actual_flows)
                    elif self.args.actor_type == 'tfn_goal_as_flow':
                        start_tool_states, end_tool_states = data_batch['start_tool_states'][i], data_batch['end_tool_states'][i]
                        mid_tool_states = data_batch['mid_tool_states'][i]
                        tool_pcl_n = transforms.quaternion_apply(end_tool_states[:, :, 3:7], tool_particles) + end_tool_states[:, :, :3]
                        tool_pcl_mid = transforms.quaternion_apply(mid_tool_states[:, :, 3:7], tool_particles) + mid_tool_states[:, :, :3]                    
                        tool_pcl = transforms.quaternion_apply(start_tool_states[:, :, 3:7], tool_particles) + start_tool_states[:, :, :3]
                        actual_flows_mid = tool_pcl_mid - tool_pcl

                        tool_pcl = torch.cat([tool_pcl, torch.zeros_like(tool_pcl).cuda()], dim=-1)
                        matched_goals = data_batch['matched_goal_obses'][i]
                        goal_flow = matched_goals - init_obses
                        init_obses = torch.cat([init_obses, goal_flow], dim=-1)
                        data_all, _ = self.organize_pc_data_tfn_goalflow([init_obses, tool_pcl], center=False)
                        pred_flows_mid, per_pt_flows_mid = self.actor(data_all)

                        pred_tool_pcl = tool_pcl[:, :, :3] + pred_flows_mid / self.actor.scale_pcl_val
                        actual_flows = tool_pcl_n - pred_tool_pcl
                        pred_tool_pcl = torch.cat([pred_tool_pcl, torch.zeros_like(pred_tool_pcl).cuda()], dim=-1)
                        data_all, _ = self.organize_pc_data_tfn_goalflow([init_obses, pred_tool_pcl], center=False)
                        pred_flows, per_pt_flows = self.actor(data_all)        
                        if self.actor.scale_pcl_val is not None:    
                            actual_flows = actual_flows * self.actor.scale_pcl_val
                            actual_flows_mid = actual_flows_mid * self.actor.scale_pcl_val
                        print('flow_magnitudes', torch.norm(actual_flows, dim=-1).mean().item(), torch.norm(pred_flows, dim=-1).mean().item())
                        print('flow_magnitudes_mid', torch.norm(actual_flows_mid, dim=-1).mean().item(), torch.norm(pred_flows_mid, dim=-1).mean().item())
                        flow_loss = self.actor.tfn_action_loss(pred_flows, per_pt_flows, actual_flows) + self.actor.tfn_action_loss(pred_flows_mid, per_pt_flows_mid, actual_flows_mid)
                    action_losses['flow_loss'] = flow_loss
                else:
                    raise NotImplementedError
            else:
                action_losses = {'action_loss': 0.}
            
            # tool value function
            if 'value_fn' in self.args.train_modules:
                other_tids = [k for k in tids if k != tid]
                if self.args.value_fn_type == 'separate':
                    val_positive, _ = self.value_fn(data_pc, data_tool_pc)
                elif self.args.value_fn_type == 'shared':
                    assert False, 'deprecated, doesnt work as well as separate'
                    assert self.args.actor_type in ['conditioned'], "shared value fn only works with conditioned actor for now"
                    with torch.no_grad():
                        pred_init_pose, _ = self.actor.init_pose_actor(data_pc, data_tool_pc)
                        pred_init_pose = pred_init_pose.view(B, -1, 6)
                        moved_tool_particles = transforms.quaternion_apply(
                                                transforms.matrix_to_quaternion(
                                                transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                                                ), tool_particles) # B x 1 x 4 apply to B x 200 x 3
                        moved_tool_particles = moved_tool_particles + pred_init_pose[:, 1:2, :3].clone() + mean
                        if self.args.actor_type == 'conditioned_goal_as_flow':
                            moved_tool_particles = torch.cat([moved_tool_particles, torch.zeros_like(moved_tool_particles).cuda()], dim=-1)
                            matched_goals = data_batch['matched_goal_obses'][i]
                            goal_flow = matched_goals - init_obses
                            init_obses = torch.cat([init_obses, goal_flow], dim=-1)
                            data_all, _ = self.organize_pc_data_tfn_goalflow([init_obses, moved_tool_particles])
                        else:
                            data_all, _ = self.organize_pc_data_scene([init_obses, moved_tool_particles, goal_obses])
                    val_positive, _ = self.value_fn(data_all)
                if len(other_tids) > 0:
                    negative_init_tool_particles = self.tool_particles[np.random.choice(other_tids)][0].tile(B, 1, 1)
                    # if self.args.random_init_pose:
                    #     random_quats2 = random_so3(B, rot_var=np.pi*2).view(B, 1, 4).cuda()
                    #     negative_init_tool_particles = transforms.quaternion_apply(random_quats2, negative_init_tool_particles)
                    negative_data_tool_pc, _ = self.organize_pc_data_pn(negative_init_tool_particles)
                    if self.args.value_fn_type == 'separate':
                        val_negative, _ = self.value_fn(data_pc, negative_data_tool_pc)
                    elif self.args.value_fn_type == 'shared':
                        assert False, 'deprecated, doesnt work as well as separate'
                        with torch.no_grad():
                            neg_pred_init_pose, _ = self.actor.init_pose_actor(data_pc, negative_data_tool_pc)
                            neg_pred_init_pose = neg_pred_init_pose.view(B, -1, 6)
                            neg_moved_tool_particles = transforms.quaternion_apply(
                                                    transforms.matrix_to_quaternion(
                                                    transforms.rotation_6d_to_matrix(neg_pred_init_pose[:, 0:1, :].clone())
                                                    ), negative_init_tool_particles) # B x 1 x 4 apply to B x 1000 x 3
                            neg_moved_tool_particles = neg_moved_tool_particles + neg_pred_init_pose[:, 1:2, :3].clone() + mean
                            negative_data_all, _ = self.organize_pc_data_scene([init_obses, neg_moved_tool_particles, goal_obses])
                        val_negative, _ = self.value_fn(negative_data_all)
                    val_preds = torch.cat([val_positive, val_negative], dim=0)
                    val_fn_loss = self.value_fn.MSELoss(val_preds, success_flag)
                else:
                    print("only training one tool so no hard negatives")
                    val_preds = val_positive
                    val_fn_loss = self.value_fn.MSELoss(val_preds, success_flag[:B])
            else:
                val_fn_loss = 0.
            d = {}
            d.update(**action_losses)
            d['val_fn_loss'] = val_fn_loss
            log_dict.log_dict(d)

        sum_dict = log_dict.agg(reduction='sum', numpy=False)
        if 'tfn' in self.args.actor_type:
            l = sum_dict['action_loss'] + sum_dict['val_fn_loss'] + sum_dict['flow_loss']
        else:
            l = sum_dict['action_loss'] + sum_dict['val_fn_loss']

        if mode == 'train':
            self.optim.zero_grad()
            l.backward()
            self.optim.step()
        ret_dict = log_dict.agg(reduction='mean', numpy=True)
        ret_dict = dict_add_prefix(ret_dict, 'avg_', skip_substr='sgld')
        return ret_dict

    def train(self, *args, **kwargs):
        return self.train_pc(*args, **kwargs), {}

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'value_fn': self.value_fn.state_dict()}, path)

    def load(self, path, modules=('policy', 'value_fn')):
        ckpt = torch.load(path)
        if 'policy' in modules:
            self.actor.load_state_dict(ckpt['actor'])
        if 'value_fn' in modules:
            self.value_fn.load_state_dict(ckpt['value_fn'])
        print('Agent {} loaded from {}'.format(modules, path))

    def load_train_stats(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.train_stats = pickle.load(f)

    def save_train_stats(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.train_stats, f)

    def update_best_model(self, epoch, eval_info):
        if not hasattr(self, 'best_actor'):
            self.best_actor = BestModel(self.actor, os.path.join(logger.get_dir(), 'best_actor.ckpt'), 'actor')
            self.best_value_fn = BestModel(self.value_fn, os.path.join(logger.get_dir(), 'best_value_fn.ckpt'), 'value_fn')
        self.best_actor.update(epoch, eval_info.get('eval/avg_action_loss_mean', 0.))
        self.best_value_fn.update(epoch, eval_info.get('eval/avg_value_loss_mean', 0.))
        self.best_dict = {'eval/best_actor_epoch': self.best_actor.best_epoch,'eval/best_value_fn': self.best_actor.best_epoch,}

    def load_best_model(self):
        # Save training model
        self.training_model_param = {
            'actor': copy.deepcopy(self.actor.state_dict()),
            'value_fn': copy.deepcopy(self.value_fn.state_dict())}
        # Use best model
        self.actor.load_state_dict(self.best_actor.param)
        self.value_fn.load_state_dict(self.best_value_fn.param)

    def load_training_model(self):
        self.actor.load_state_dict(self.training_model_param['actor'])
        self.value_fn.load_state_dict(self.training_model_param['value_fn'])

class BestModel(object):
    def __init__(self, model, save_path, name):
        self.model = model
        self.save_path = save_path
        self.name = name

    def update(self, epoch, loss):
        if not hasattr(self, 'best_loss') or (loss < self.best_loss):
            self.best_loss = loss
            self.best_epoch = epoch
            self.param = copy.deepcopy(self.model.state_dict())
            torch.save({self.name: self.param}, self.save_path)
