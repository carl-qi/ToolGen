import torch
import torch.nn as nn
import torch.nn.functional as F
from core.toolgen.model import PointNet2Segmentation, ActorEncoder

class ActionScore(nn.Module):
    def __init__(self, feat_dim, task_feat_dim=32, cp_feat_dim=32, topk=5):
        super(ActionScore, self).__init__()

        self.z_dim = feat_dim
        self.topk = topk


        self.mlp = nn.ModuleList([nn.Linear(feat_dim + task_feat_dim + cp_feat_dim, feat_dim), 
        nn.Linear(feat_dim, feat_dim), 
        nn.Linear(feat_dim, feat_dim),
        nn.Linear(feat_dim, feat_dim),
        nn.Linear(feat_dim, 1)])

    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, whole_feats: B x F x N
    def forward(self, pcs, task, contact_point):
        # pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]
        f_ctpt = self.mlp_cp(contact_point)
        task = task.view(-1, 1)
        f_task = self.mlp_task(task)
        score = torch.sigmoid(self.mlp(net, f_task, f_ctpt))

        return score

    def inference_action_score(self, pcs, task):
        # pcs[:, 0] = contact_point
        batch_size = pcs.shape[0]
        pt_size = pcs.shape[1]
        contact_point = pcs.view(batch_size * pt_size, -1)
        f_ctpt = self.mlp_cp(contact_point)
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net = whole_feats.permute(0, 2, 1).reshape(batch_size*pt_size, -1)

        task = task.view(-1, 1)
        f_task = self.mlp_task(task)
        f_task = f_task.repeat(pt_size, 1)
        score = torch.sigmoid(self.mlp(net, f_task, f_ctpt))

        return score

    def get_loss(self, pcs, task, contact_point, actor, critic, rvs=100):
        batch_size = pcs.shape[0]
        with torch.no_grad():
            traj = actor.sample_n(pcs, task, contact_point, rvs=rvs)
        with torch.no_grad():
            gt_score = torch.sigmoid(critic.forward_n(pcs, task, traj, contact_point, rvs=rvs)[0])
            gt_score = gt_score.view(batch_size, rvs, 1).topk(k=self.topk, dim=1)[0].mean(dim=1).view(-1)
        score = self.forward(pcs, task, contact_point)
        loss = self.L1Loss(score.view(-1), gt_score).mean()

        return loss

    def inference_whole_pc(self, feats, dirs1, dirs2):
        num_pts = feats.shape[-1]
        batch_size = feats.shape[0]

        feats = feats.permute(0, 2, 1)  # B x N x F
        feats = feats.reshape(batch_size*num_pts, -1)

        input_queries = torch.cat([dirs1, dirs2], dim=-1)
        input_queries = input_queries.unsqueeze(dim=1).repeat(1, num_pts, 1)
        input_queries = input_queries.reshape(batch_size*num_pts, -1)

        pred_result_logits = self.critic(feats, input_queries)

        soft_pred_results = torch.sigmoid(pred_result_logits)
        soft_pred_results = soft_pred_results.reshape(batch_size, num_pts)

        return soft_pred_results

    def inference(self, pcs, dirs1, dirs2):
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]

        input_queries = torch.cat([dirs1, dirs2], dim=1)

        pred_result_logits = self.critic(net, input_queries)

        pred_results = (pred_result_logits > 0)

        return pred_results
