import torch
import torch.nn as nn
from core.diffskill import utils
from core.diffskill.agents.pointnet_encoder import PointNetEncoder, PointNetEncoder2, PointNetEncoderCat
from core.toolgen.se3 import flow2pose_tfn
from core.utils.pcl_visualization_utils import visualize_point_cloud
from core.toolgen.model import PointNet2Segmentation
from torch_geometric.nn import MLP
from core.toolgen.utils import get_6d_rot_loss
import pytorch3d.transforms as transforms

class PointActorCat(nn.Module):
    """ PointCloud based actor with tool state concatenation"""

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoderCat(2)
        hidden_dim = args.actor_latent_dim
        self.dimu, self.dimtool = args.dimu, args.dimtool
        self.state_encoder = nn.Sequential(nn.Linear(self.dimtool, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, 16))
        self.mlp = nn.Sequential(nn.Linear(1024+16, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(1024+16, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 1))

        self.outputs = dict()
        self.loss = torch.nn.MSELoss(reduction='none')
        self.apply(utils.weight_init)

    def forward(self, data, detach_encoder=False):
        h_dough = self.encoder(data, detach=detach_encoder)
        h_tool = self.state_encoder(data['s_tool'])
        h = torch.cat([h_dough, h_tool], dim=-1)
        action = self.mlp(h)
        done = self.done_mlp(h)
        # hardcode for good initialization
        action = action / 5.
        done = done / 5.
        return action, done

class PointActor(nn.Module):
    """ PointCloud based actor"""

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(input_dim)
        self.tool_encoder = PointNetEncoder2(0, out_dim=256)
        self.mlp = nn.Sequential(nn.Linear(1024+256, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, action_dim))
        # self.done_mlp = nn.Sequential(nn.Linear(1024, 512),
        #                               nn.ReLU(),
        #                               nn.Linear(512, 256),
        #                               nn.ReLU(),
        #                               nn.Linear(256, 1))

        self.outputs = dict()
        self.MSELoss = torch.nn.MSELoss(reduction='mean')
        self.L1Loss = torch.nn.L1Loss(reduction='mean')
        self.BCELoss= nn.BCEWithLogitsLoss(reduction='mean')
        self.apply(utils.weight_init)

    def forward(self, data, tool_data, detach_encoder=False):

        h = self.encoder(data, detach=detach_encoder)
        h_tool = self.tool_encoder(tool_data, detach=detach_encoder)
        # print(h, h_tool)
        # print(h_tool) 
        # print(h.shape, h_tool.shape)
        # print(torch.norm(h, dim=1), torch.norm(h_tool, dim=1))
            # for tiling
            # n_rep = h.shape[-1] // h_tool.shape[-1]
            # h = torch.cat([h, h_tool.repeat(1, n_rep)], dim=-1)
        h = torch.cat([h, h_tool], dim=-1)
        action = self.mlp(h)
        # done = self.done_mlp(h)
        # hardcode for good initialization
        # action = action / 5.
        # done = done / 5.
        return action, h
    
    def traj_action_loss(self, recon_traj, traj):
        batch_size = traj.shape[0]
        recon_traj = recon_traj.view(batch_size, self.args.horizon, 6)
        traj = traj.view(batch_size, self.args.horizon, 6)
        recon_dir = recon_traj[:, 0, :]
        recon_wps = recon_traj[:, 1:, :]
        input_dir = traj[:, 0, :]
        input_wps = traj[:, 1:, :]
        recon_loss = self.L1Loss(recon_wps.view(batch_size, (self.args.horizon - 1) * 6), input_wps.view(batch_size, (self.args.horizon - 1) * 6))

        dir_loss = get_6d_rot_loss(recon_dir, input_dir)
        dir_loss = dir_loss.mean()

        losses = {}
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['action_loss'] = recon_loss + dir_loss
        return losses

class PointActorRecurrent(nn.Module):
    """ PointCloud based actor, with gru"""

    def __init__(self, args, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.args = args
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.init_pose_actor = PointActor(args, input_dim=2, action_dim=2 * 6)
        self.encoder = PointNetEncoder(input_dim, out_dim=512)
        self.mlp = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, hidden_dim))
        self.gru = nn.GRU(hidden_size=hidden_dim, input_size=6, batch_first=True)
        self.out_mlp = nn.Sequential(nn.Linear(hidden_dim, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, action_dim))
        self.outputs = dict()
        self.MSELoss = torch.nn.MSELoss(reduction='mean')
        self.L1Loss = torch.nn.L1Loss(reduction='mean')
        self.BCELoss= nn.BCEWithLogitsLoss(reduction='mean')
        self.apply(utils.weight_init)

    def forward(self, data, horizon, detach_encoder=False):
        h = self.encoder(data, detach=detach_encoder)
        B = h.shape[0]
        gru_h = self.mlp(h).view(1, B, self.hidden_dim)
        gru_input = torch.zeros(B, 1, self.action_dim).cuda()
        delta_ws = []
        ws = []
        for t in range(horizon):
            gru_output, gru_h = self.gru(gru_input, gru_h)
            delta_w = self.out_mlp(gru_output)
            delta_ws.append(delta_w)
            gru_input = gru_input + delta_w
            ws.append(gru_input)
        
        return torch.cat(delta_ws, dim=1), torch.cat(ws, dim=1)
    
    def traj_action_loss(self, pred_init_pose, pred_traj, tool_traj):
        B = tool_traj.shape[0]

        # init_rot = transforms.quaternion_multiply(
        #     tool_traj[:, 1, 3:7],
        #     transforms.quaternion_invert(tool_traj[:, 0, 3:7])
        # )
        init_rot = transforms.quaternion_to_matrix(tool_traj[:, 1, 3:7])
        init_rot = transforms.matrix_to_rotation_6d(init_rot).view(B, 1, 6)  # B x 1 x 6
        init_pos = tool_traj[:, 1, :3].view(B, 1, 3)
        init_pos = init_pos.repeat(1, 1, 2)  # B x 1 x 6
        dir_loss = get_6d_rot_loss(init_rot, pred_init_pose[:, 0:1, :]) # B x 1 x 6
        dir_loss = dir_loss.mean()
        
        recon_loss = self.L1Loss(init_pos, pred_init_pose[:, 1:2, :])
        pos_label = tool_traj[:, 2:, :3] - pred_init_pose[:, 1:2, :3]  # B x horizon-2 x 3
        rotations_n = tool_traj[:, 2:, 3:7] # B x horizon-2 x 4
        rotations_inv = transforms.quaternion_invert(
                                        transforms.matrix_to_quaternion(
                                        transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                                        )) # B x 1 x 4
        delta_rotations = transforms.quaternion_multiply(rotations_n, rotations_inv)
        angles_label = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(delta_rotations), 'XYZ')  # B x horizon-2 x 3
        traj_label = torch.cat([pos_label, angles_label], dim=-1)
        recon_loss += self.L1Loss(pred_traj, traj_label)

        losses = {}
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['action_loss'] = recon_loss + dir_loss
        return losses

class PointActorConditioned(nn.Module):
    """ PointCloud based actor, taking the entire point cloud of the scene as input"""

    def __init__(self, args, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.args = args
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.init_pose_actor = PointActor(args, input_dim=2, action_dim=2 * 6)
        self.encoder = PointNetEncoder(input_dim, out_dim=512)
        self.mlp = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, action_dim))
        self.outputs = dict()
        self.MSELoss = torch.nn.MSELoss(reduction='mean')
        self.L1Loss = torch.nn.L1Loss(reduction='mean')
        self.BCELoss= nn.BCEWithLogitsLoss(reduction='mean')
        self.apply(utils.weight_init)

    def forward(self, data, detach_encoder=False):
        h = self.encoder(data, detach=detach_encoder)
        B = h.shape[0]
        if self.action_dim % 6 == 0:
            action = self.mlp(h).view(B, -1, 6)
        else:
            action = self.mlp(h)
        return action, h
    
    def traj_action_loss(self, pred_init_pose, pred_deltas, tool_traj):
        B = tool_traj.shape[0]

        # init_rot = transforms.quaternion_multiply(
        #     tool_traj[:, 1, 3:7],
        #     transforms.quaternion_invert(tool_traj[:, 0, 3:7])
        # )
        init_rot = transforms.quaternion_to_matrix(tool_traj[:, 1, 3:7])
        init_rot = transforms.matrix_to_rotation_6d(init_rot).view(B, 1, 6)  # B x 1 x 6
        init_pos = tool_traj[:, 1, :3].view(B, 1, 3)
        init_pos = init_pos.repeat(1, 1, 2)  # B x 1 x 6
        dir_loss = get_6d_rot_loss(init_rot, pred_init_pose[:, 0:1, :]) # B x 1 x 6
        dir_loss = dir_loss.mean()
        
        recon_loss = self.L1Loss(init_pos, pred_init_pose[:, 1:2, :])
        pos_label = tool_traj[:, 2:, :3] - pred_init_pose[:, 1:2, :3]  # B x horizon-2 x 3
        rotations_n = tool_traj[:, 2:, 3:7] # B x horizon-2 x 4
        rotations_inv = transforms.quaternion_invert(
                                        transforms.matrix_to_quaternion(
                                        transforms.rotation_6d_to_matrix(pred_init_pose[:, 0:1, :].clone())
                                        )) # B x 1 x 4
        delta_rotations = transforms.quaternion_multiply(rotations_n, rotations_inv)
        angles_label = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(delta_rotations), 'XYZ')  # B x horizon-2 x 3
        traj_label = torch.cat([pos_label, angles_label], dim=-1)

        pred_traj = torch.zeros_like(pred_deltas).cuda()
        pred_traj[:, 0, :] = pred_deltas[:, 0, :]
        for t in range(1, pred_deltas.shape[1]):
            pred_traj[:, t, :] = pred_traj[:, t-1, :] + pred_deltas[:, t, :]
        recon_loss += self.L1Loss(pred_traj, traj_label)

        losses = {}
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['action_loss'] = recon_loss + dir_loss
        return losses

class PointActorTFNTraj(nn.Module):
    """ PointCloud based actor, with gru"""

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.flow_dim = action_dim
        self.scale_pcl_val = 50
        self.init_pose_actor = PointActor(args, input_dim=2, action_dim=2 * 6)
        self.encoder = PointNet2Segmentation(input_dim, 0)
        self.mlp = MLP([128, 128, 128, self.flow_dim], dropout=0.5, batch_norm=False)
        self.MSELoss = torch.nn.MSELoss(reduction='mean')
        self.L1Loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, data, detach_encoder=False):
        x = self.encoder(data, detach=detach_encoder)
        flow_per_pt = self.mlp(x)
        # Must revert `data.pos` back to the original scale before SVD!!
        if self.scale_pcl_val is not None:
            data['pos'] = data['pos'] * self.scale_pcl_val

        batch_flow_traj = []
        batch_before_svd_flow_traj = []
        B = len(data['ptr'])-1
        T = self.flow_dim // 3
        for i in range(B):
            flow_traj = torch.zeros((1, T, 1000, 3)).cuda()
            before_svd_flow_traj = torch.zeros((1, T, 1000, 3)).cuda()
            idx1 = data['ptr'][i].detach().cpu().numpy().item()
            idx2 = data['ptr'][i+1].detach().cpu().numpy().item()
            posi_one = data['pos'][idx1:idx2]  # just this PCL's xyz (tool+items).
            tool_one = torch.where(data['x'][idx1:idx2, -2] == 1)[0] # tool 0-th col
            xyz  = posi_one[tool_one]
            for t in range(T):
                flow_one = flow_per_pt[idx1:idx2, t*3:(t+1)*3]  # just this PCL's flow.
                flow = flow_one[tool_one]

                trfm, _, _ = flow2pose_tfn(
                            xyz=xyz[None,:],
                            flow=flow[None,:],
                            weights=None,
                            world_frameify=True
                    )

                trfm_xyz = trfm.transform_points(xyz).squeeze(0)
                batch_flow = trfm_xyz - xyz
                flow_traj[:, t, :] = batch_flow
                before_svd_flow_traj[:, t, :] = flow
                xyz = trfm_xyz.clone()

            batch_flow_traj.append(flow_traj)
            batch_before_svd_flow_traj.append(before_svd_flow_traj)
        if self.scale_pcl_val is not None:
            data['pos'] = data['pos'] / self.scale_pcl_val
        return torch.vstack(batch_flow_traj).view(B, T, -1, 3), torch.vstack(batch_before_svd_flow_traj).view(B, T, -1, 3)

    def get_delta_pose_xyz_flow(self, xyz, flow):
        # x = self.encoder(data)
        # flow_per_pt = self.mlp(x)
        # # Must revert `data.pos` back to the original scale before SVD!!
        if self.scale_pcl_val is not None:
            xyz = xyz * self.scale_pcl_val
        #     data['pos'] = data['pos'] * self.scale_pcl_val

        # idx1 = data['ptr'][0].detach().cpu().numpy().item()
        # idx2 = data['ptr'][1].detach().cpu().numpy().item()
        # flow_one = flow_per_pt[idx1:idx2]  # just this PCL's flow.
        # posi_one = data['pos'][idx1:idx2]  # just this PCL's xyz (tool+items).
        # tool_one = torch.where(data['x'][idx1:idx2, -2] == 1)[0] # tool 0-th col
        # xyz  = posi_one[tool_one]
        # flow = flow_one[tool_one]
        trfm, R, t = flow2pose_tfn(
                    xyz=xyz[None,:],
                    flow=flow[None,:],
                    weights=None,
                    world_frameify=False
            )
        if self.scale_pcl_val is not None:
            xyz = xyz / self.scale_pcl_val
        #     data['pos'] = data['pos'] / self.scale_pcl_val
        return R, t

    def reset_action_loss(self, pred_init_pose, tool_traj):
        B = tool_traj.shape[0]

        init_rot = transforms.quaternion_to_matrix(tool_traj[:, 1, 3:7])
        init_rot = transforms.matrix_to_rotation_6d(init_rot).view(B, 1, 6)  # B x 1 x 6
        init_pos = tool_traj[:, 1, :3].view(B, 1, 3)
        init_pos = init_pos.repeat(1, 1, 2)  # B x 1 x 6
        dir_loss = get_6d_rot_loss(init_rot, pred_init_pose[:, 0:1, :]) # B x 1 x 6
        dir_loss = dir_loss.mean()
        recon_loss = self.L1Loss(init_pos, pred_init_pose[:, 1:2, :])
        losses = {}
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['action_loss'] = recon_loss + dir_loss
        return losses

    def tfn_action_loss(self, predicted_flows, per_pt_flows, actual_flows):
        action_loss = self.MSELoss(predicted_flows, actual_flows)
        consistency_loss = self.MSELoss(predicted_flows, per_pt_flows)
        action_loss += 0.1 * consistency_loss
        return action_loss




class PointActorTFN(nn.Module):
    """ PointCloud based actor, with gru"""

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.flow_dim = action_dim
        self.scale_pcl_val = 50
        self.init_pose_actor = PointActor(args, input_dim=2, action_dim=2 * 6)
        self.encoder = PointNet2Segmentation(input_dim, 0)
        self.mlp = MLP([128, 128, 128, self.flow_dim], dropout=0.5, batch_norm=False)
        self.MSELoss = torch.nn.MSELoss(reduction='mean')
        self.L1Loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, data, detach_encoder=False):
        x = self.encoder(data, detach=detach_encoder)
        flow_per_pt = self.mlp(x)
        # Must revert `data.pos` back to the original scale before SVD!!
        if self.scale_pcl_val is not None:
            data['pos'] = data['pos'] * self.scale_pcl_val

        batch_flows = []
        before_svd_flows = []
        B = len(data['ptr'])-1
        for i in range(B):
            idx1 = data['ptr'][i].detach().cpu().numpy().item()
            idx2 = data['ptr'][i+1].detach().cpu().numpy().item()
            flow_one = flow_per_pt[idx1:idx2]  # just this PCL's flow.
            posi_one = data['pos'][idx1:idx2]  # just this PCL's xyz (tool+items).
            tool_one = torch.where(data['x'][idx1:idx2, -2] == 1)[0] # tool 0-th col
            xyz  = posi_one[tool_one]
            flow = flow_one[tool_one]

            trfm, _, _ = flow2pose_tfn(
                        xyz=xyz[None,:],
                        flow=flow[None,:],
                        weights=None,
                        world_frameify=True
                )

            trfm_xyz = trfm.transform_points(xyz).squeeze(0)
            batch_flow = trfm_xyz - xyz
            batch_flows.append(batch_flow)
            before_svd_flows.append(flow)
        if self.scale_pcl_val is not None:
            data['pos'] = data['pos'] / self.scale_pcl_val
        return torch.cat(batch_flows).view(B, -1, 3), torch.cat(before_svd_flows).view(B, -1, 3)

    def get_delta_pose(self, data):
        x = self.encoder(data)
        flow_per_pt = self.mlp(x)
        # Must revert `data.pos` back to the original scale before SVD!!
        if self.scale_pcl_val is not None:
            data['pos'] = data['pos'] * self.scale_pcl_val

        idx1 = data['ptr'][0].detach().cpu().numpy().item()
        idx2 = data['ptr'][1].detach().cpu().numpy().item()
        flow_one = flow_per_pt[idx1:idx2]  # just this PCL's flow.
        posi_one = data['pos'][idx1:idx2]  # just this PCL's xyz (tool+items).
        tool_one = torch.where(data['x'][idx1:idx2, -2] == 1)[0] # tool 0-th col
        xyz  = posi_one[tool_one]
        flow = flow_one[tool_one]
        trfm, R, t = flow2pose_tfn(
                    xyz=xyz[None,:],
                    flow=flow[None,:],
                    weights=None,
                    world_frameify=False
            )
        if self.scale_pcl_val is not None:
            data['pos'] = data['pos'] / self.scale_pcl_val
        return R, t

    def reset_action_loss(self, pred_init_pose, tool_traj):
        B = tool_traj.shape[0]

        init_rot = transforms.quaternion_to_matrix(tool_traj[:, 1, 3:7])
        init_rot = transforms.matrix_to_rotation_6d(init_rot).view(B, 1, 6)  # B x 1 x 6
        init_pos = tool_traj[:, 1, :3].view(B, 1, 3)
        init_pos = init_pos.repeat(1, 1, 2)  # B x 1 x 6
        dir_loss = get_6d_rot_loss(init_rot, pred_init_pose[:, 0:1, :]) # B x 1 x 6
        dir_loss = dir_loss.mean()
        recon_loss = self.L1Loss(init_pos, pred_init_pose[:, 1:2, :])
        losses = {}
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['action_loss'] = recon_loss + dir_loss
        return losses

    def tfn_action_loss(self, predicted_flows, per_pt_flows, actual_flows):
        action_loss = self.MSELoss(predicted_flows, actual_flows)
        consistency_loss = self.MSELoss(predicted_flows, per_pt_flows)
        action_loss += 0.1 * consistency_loss
        return action_loss

class MLPActor(nn.Module):
    """ MLP actor taking in point features as input"""

    def __init__(self, args, in_dim, action_dim):
        super().__init__()
        self.args = args
        self.mlp = nn.Sequential(nn.Linear(in_dim, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, action_dim))
        # self.done_mlp = nn.Sequential(nn.Linear(1024, 512),
        #                               nn.ReLU(),
        #                               nn.Linear(512, 256),
        #                               nn.ReLU(),
        #                               nn.Linear(256, 1))

        self.outputs = dict()
        self.loss = torch.nn.MSELoss(reduction='none')
        self.L1Loss = torch.nn.L1Loss(reduction='mean')
        self.BCELoss= nn.BCEWithLogitsLoss(reduction='mean')
        self.apply(utils.weight_init)

    def forward(self, h, detach_encoder=False):

        action = self.mlp(h)
        # done = self.done_mlp(h)
        # hardcode for good initialization
        # action = action / 5.
        # done = done / 5.
        return action, None
    
    def traj_action_loss(self, recon_traj, traj):
        batch_size = traj.shape[0]
        recon_traj = recon_traj.view(batch_size, self.args.horizon, 6)
        traj = traj.view(batch_size, self.args.horizon, 6)
        recon_dir = recon_traj[:, 0, :]
        recon_wps = recon_traj[:, 1:, :]
        input_dir = traj[:, 0, :]
        input_wps = traj[:, 1:, :]
        recon_loss = self.L1Loss(recon_wps.view(batch_size, (self.args.horizon - 1) * 6), input_wps.view(batch_size, (self.args.horizon - 1) * 6))

        dir_loss = get_6d_rot_loss(recon_dir, input_dir)
        dir_loss = dir_loss.mean()

        losses = {}
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['action_loss'] = recon_loss + dir_loss
        return losses
