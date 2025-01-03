import torch
import torch.nn as nn
import torch.nn.functional as F
from core.diffskill.agents.pointnet_encoder import SAModule, GlobalSAModule, FPModule, MLP

def KL(mu, logvar):
    mu = mu.view(mu.shape[0], -1)
    logvar = logvar.view(logvar.shape[0], -1)
    loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
    # high star implementation
    # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
    loss = torch.mean(loss)
    return loss

class PointNet2Segmentation(torch.nn.Module):
    """PointNet++, pointwise prediction"""
    def __init__(self, in_dim, feat_dim):
        super(PointNet2Segmentation, self).__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + in_dim, 128, 128, 128]))

        # self.fc_layer = nn.Sequential(
        #     # nn.Conv1d(128, 128, kernel_size=1, bias=False),
        #     # nn.BatchNorm1d(128),
        #     nn.Linear(128, 128),
        #     nn.ReLU(True),
        # )

    def forward(self, data, detach=False):
        sa0_out = (data['x'], data['pos'], data['batch'])
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        # x = self.fc_layer(x)

        if detach:
            x.detach()
        return x

class PointNet2SegmentationGlobal(torch.nn.Module):
    """PointNet++, pointwise prediction"""
    def __init__(self, in_dim, feat_dim):
        super(PointNet2SegmentationGlobal, self).__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + in_dim, 128, 128, feat_dim]))

        # self.fc_layer = nn.Sequential(
        #     # nn.Conv1d(128, 128, kernel_size=1, bias=False),
        #     # nn.BatchNorm1d(128),
        #     nn.Linear(128, 128),
        #     nn.ReLU(True),
        # )

    def forward(self, data, detach=False):
        sa0_out = (data['x'], data['pos'], data['batch'])
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        # x = self.fc_layer(x)

        if detach:
            x.detach()
        return x, sa3_out[0]


class ActorEncoder(nn.Module):
    def __init__(self, feat_dim, traj_z_dim=128, cp_feat_dim=32, num_steps=10):
        super(ActorEncoder, self).__init__()

        self.mlp1 = nn.Linear(feat_dim + traj_z_dim + cp_feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, feat_dim)
        self.get_mu = nn.Linear(feat_dim, feat_dim)
        self.get_logvar = nn.Linear(feat_dim, feat_dim)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, traj_feats, pixel_feats, f_ctpt):
        net = torch.cat([traj_feats, pixel_feats, f_ctpt], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net)
        mu = self.get_mu(net)
        logvar = self.get_logvar(net)
        noise = torch.Tensor(torch.randn(*mu.shape)).cuda()
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar

class ActorDecoder(nn.Module):
    def __init__(self, feat_dim, traj_z_dim=128, cp_feat_dim=32, num_steps=10):
        super(ActorDecoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + traj_z_dim + cp_feat_dim, 512),
            nn.Linear(512, 256),
            nn.Linear(256, num_steps * 6)
        )
        self.num_steps = num_steps

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, z_all, f_p, f_ctpt):
        batch_size = z_all.shape[0]
        x = torch.cat([z_all, f_p, f_ctpt], dim=-1)
        x = self.mlp(x)
        x = x.view(batch_size, self.num_steps, 6)
        return x

class TrajEncoder(nn.Module):
    def __init__(self, traj_feat_dim=128, num_steps=50):
        super(TrajEncoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_steps * 6, 128),
            nn.Linear(128, 128),
            nn.Linear(128, traj_feat_dim)
        )

        self.num_steps = num_steps

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x.view(batch_size, self.num_steps * 6))
        return x

# class Critic(nn.Module):
#     def __init__(self, feat_dim, task_feat_dim=32, traj_feat_dim=256, cp_feat_dim=32, num_steps=10):
#         super(Critic, self).__init__()

#         self.mlp1 = nn.Linear(feat_dim + traj_feat_dim + task_feat_dim + cp_feat_dim, feat_dim)
#         self.mlp2 = nn.Linear(feat_dim, 1)
#         self.num_steps = num_steps

#     # pixel_feats B x F, query_fats: B x 6
#     # output: B
#     def forward(self, pixel_feats, task, traj, contact_point, encoder_modules=[]):
#         mlp_cp, mlp_traj, mlp_task = encoder_modules
#         batch_size = traj.shape[0]
#         task = task.view(-1, 1)
#         traj = traj.view(batch_size, self.num_steps * 6)
#         task_feat = mlp_task(task)
#         traj_feats = mlp_traj(traj)
#         cp_feats = mlp_cp(contact_point)
#         net = torch.cat([pixel_feats, task_feat, traj_feats, cp_feats], dim=-1)
#         net = F.leaky_relu(self.mlp1(net))
#         net = self.mlp2(net).squeeze(-1)
#         return net

#     def forward_n(self, pixel_feats, task, traj, contact_point, rvs, encoder_modules=[]):
#         mlp_cp, mlp_traj, mlp_task = encoder_modules
#         batch_size = pixel_feats.shape[0]
#         task = task.view(-1, 1)
#         traj = traj.view(batch_size * rvs, self.num_steps * 6)
#         task_feat = mlp_task(task)
#         traj_feats = mlp_traj(traj)
#         cp_feats = mlp_cp(contact_point)
#         pixel_feats = pixel_feats.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
#         task_feat = task_feat.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
#         cp_feats = cp_feats.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
#         net = torch.cat([pixel_feats, task_feat, traj_feats, cp_feats], dim=-1)
#         net = F.leaky_relu(self.mlp1(net))
#         net = self.mlp2(net).squeeze(-1)
#         return net

#     # cross entropy loss
#     def get_ce_loss(self, pred_logits, gt_labels):
#         loss = self.BCELoss(pred_logits, gt_labels.float())
#         return loss

#     # cross entropy loss
#     def get_l1_loss(self, pred_logits, gt_labels):
#         loss = self.L1Loss(pred_logits, gt_labels)
#         return loss