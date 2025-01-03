import torch
import torch.nn as nn
import torch.nn.functional as F

# from core.toolgen.model import Critic

class TrajScore(nn.Module):
    def __init__(self, args):
        super(TrajScore, self).__init__()

        self.args = args
        self.mlp1 = nn.Linear(args.feat_dim + args.traj_feat_dim + args.cp_feat_dim, args.feat_dim)
        self.mlp2 = nn.Linear(args.feat_dim, 1)
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='mean')
        self.L1Loss = nn.L1Loss(reduction='none')
        self.num_steps = self.args.horizon

    # pred_result_logits: B
    def forward(self, f_p, f_s, f_tau):

        net = torch.cat([f_s, f_tau, f_p], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net)
        return net


    def forward_n(self, pcs, task, traj, contact_point, rvs):
        ### DEPRECATED
        raise NotImplementedError

        # pcs[:, 0] = contact_point
        batch_size = task.shape[0]
        pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]

        # input_queries = torch.cat([dirs1, dirs2], dim=1)

        pred_result_logits = self.critic.forward_n(net, task, traj, contact_point, rvs=rvs)

        return pred_result_logits, whole_feats

    def inference_critic_score(self, pcs, task, traj):
        ### DEPRECATED
        raise NotImplementedError

        # pcs[:, 0] = contact_point
        batch_size = pcs.shape[0]
        pt_size = pcs.shape[1]
        contact_point = pcs.view(batch_size * pt_size, -1)
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net = whole_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)

        task = task.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, 1)

        traj = traj.repeat(batch_size * pt_size, 1, 1)

        pred_result_logits = self.critic.forward(net, task, traj, contact_point)
        pred_result_logits = torch.sigmoid(pred_result_logits)

        return pred_result_logits