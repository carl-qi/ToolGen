import torch
import torch.nn as nn
import torch.nn.functional as F

from core.toolgen.model import ActorEncoder, ActorDecoder, KL

class TrajProposal(nn.Module):
    def __init__(self, args):
        super(TrajProposal, self).__init__()

        self.args = args
        self.traj_encoder = ActorEncoder(self.args.feat_dim, traj_z_dim=self.args.traj_z_dim, num_steps=self.args.horizon)
        self.traj_decoder = ActorDecoder(self.args.feat_dim, traj_z_dim=self.args.traj_z_dim, num_steps=self.args.horizon)

        self.num_steps = self.args.horizon
        self.lbd_kl = self.args.lbd_kl
        self.lbd_recon = self.args.lbd_recon
        self.lbd_dir = self.args.lbd_dir

    # input sz bszx3x2
    def bgs(self, d6s):
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
        a2 = d6s[:, :, 1]
        b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

    # batch geodesic loss for rotation matrices
    def bgdR(self, Rgts, Rps):
        Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
        Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
        return torch.acos(theta) # / 2

    # 6D-Rot loss
    # input sz bszx6
    def get_6d_rot_loss(self, pred_6d, gt_6d):
        pred_Rs = self.bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        gt_Rs = self.bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        theta = self.bgdR(gt_Rs, pred_Rs)
        return theta

    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, whole_feats: B x F x N
    def forward(self, pcs, task, traj, contact_point):
        # pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]
        f_ctpt = self.mlp_cp(contact_point)
        task = task.view(-1, 1)
        f_task = self.mlp_task(task)

        z_traj = self.traj_encoder(traj)
        z_all, mu, logvar = self.all_encoder(z_traj, net, f_task, f_ctpt)
        recon_traj = self.decoder(z_all, net, f_task, f_ctpt)

        return recon_traj, mu, logvar

    def sample(self, pcs, task, contact_point):
        batch_size = task.shape[0]
        f_task = self.mlp_task(task)
        f_ctpt = self.mlp_cp(contact_point)
        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).cuda()

        pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]

        recon_traj = self.decoder(z_all, net, f_task, f_ctpt)
        recon_dir = recon_traj[:, 0, :]
        recon_dir = recon_dir.reshape(-1, 3, 2)
        recon_dir = self.bgs(recon_dir)
        recon_wps = recon_traj[:, 1:, :]

        return recon_dir, recon_wps

    def sample_n(self, pcs, task, contact_point, rvs=100):
        batch_size = task.shape[0]
        task = task.view(-1, 1)
        f_task = self.mlp_task(task)
        f_ctpt = self.mlp_cp(contact_point)
        z_all = torch.Tensor(torch.randn(batch_size * rvs, self.z_dim)).cuda()

        pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]
        net = net.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
        f_task = f_task.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
        f_ctpt = f_ctpt.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)

        recon_traj = self.decoder(z_all, net, f_task, f_ctpt)

        return recon_traj

    def get_loss(self, pcs, task, traj, contact_point):
        batch_size = traj.shape[0]
        recon_traj, mu, logvar = self.forward(pcs, task, traj, contact_point)
        recon_dir = recon_traj[:, 0, :]
        recon_wps = recon_traj[:, 1:, :]
        input_dir = traj[:, 0, :]
        input_wps = traj[:, 1:, :]
        recon_loss = self.L1Loss(recon_wps.view(batch_size, (self.num_steps - 1) * 6), input_wps.view(batch_size, (self.num_steps - 1) * 6))
        recon_loss = recon_loss.mean()

        dir_loss = self.get_6d_rot_loss(recon_dir, input_dir)
        dir_loss = dir_loss.mean()
        kl_loss = KL(mu, logvar)
        losses = {}
        losses['kl'] = kl_loss
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['tot'] = kl_loss * self.lbd_kl + recon_loss * self.lbd_recon + dir_loss * self.lbd_dir

        return losses

    def get_loss_test_rotation(self, pcs, task, traj, contact_point):
        batch_size = traj.shape[0]
        recon_traj, mu, logvar = self.forward(pcs, task, traj, contact_point)
        recon_dir = recon_traj[:, 0, :]
        recon_wps = recon_traj[:, 1:, :]
        input_dir = traj[:, 0, :]
        input_wps = traj[:, 1:, :]
        recon_xyz_loss = self.L1Loss(recon_wps[:, :, 0:3].contiguous().view(batch_size, (self.num_steps - 1) * 3), input_wps[:, :, 0:3].contiguous().view(batch_size, (self.num_steps - 1) * 3))
        recon_rotation_loss = self.L1Loss(recon_wps[:, :, 3:6].contiguous().view(batch_size, (self.num_steps - 1) * 3), input_wps[:, :, 3:6].contiguous().view(batch_size, (self.num_steps - 1) * 3))
        recon_loss = recon_xyz_loss.mean() + recon_rotation_loss.mean() * 100

        dir_loss = self.get_6d_rot_loss(recon_dir, input_dir)
        dir_loss = dir_loss.mean()
        kl_loss = KL(mu, logvar)
        losses = {}
        losses['kl'] = kl_loss
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['recon_xyz'] = recon_xyz_loss.mean()
        losses['recon_rotation'] = recon_rotation_loss.mean()
        losses['tot'] = kl_loss * self.lbd_kl + recon_loss * self.lbd_recon + dir_loss * self.lbd_dir

        return losses

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

