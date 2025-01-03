import torch
import torch.nn as nn
import torch.nn.functional as F
from core.toolgen.se3 import dense_flow_loss

mse_criterion = nn.MSELoss(reduction="sum")


class BrianChuerLoss(nn.Module):
    def forward(self, action_pos, T_pred, T_gt, Fx):

        pred_flow_action = (T_gt.transform_points(action_pos) - action_pos).detach()

        loss = brian_chuer_loss(
            pred_T_action=T_pred,
            gt_T_action=T_gt,
            points_trans_action=action_pos,
            pred_flow_action=Fx,
            points_action=T_gt.transform_points(action_pos),
        )
        return loss


def brian_chuer_loss(
    pred_T_action,
    gt_T_action,
    points_trans_action,
    pred_flow_action,
    points_action,
    action_weight=1.0,
    smoothness_weight=0.1,
    consistency_weight=1.0,
):
    induced_flow_action = (
        pred_T_action.transform_points(points_trans_action) - points_trans_action
    ).detach()
    pred_points_action = pred_T_action.transform_points(
        points_trans_action
    )  # pred_points_action =  T0^-1@points_trans_action

    # pred_T_action=T0^-1
    # gt_T_action = T0.inverse()

    point_loss_action = mse_criterion(pred_points_action, points_action)

    point_loss = action_weight * point_loss_action

    dense_loss = dense_flow_loss(
        points=points_trans_action, flow_pred=pred_flow_action, trans_gt=gt_T_action
    )

    # Loss associated flow vectors matching a consistent rigid transform
    smoothness_loss_action = mse_criterion(
        pred_flow_action,
        induced_flow_action,
    )

    smoothness_loss = action_weight * smoothness_loss_action

    loss = (
        point_loss
        + smoothness_weight * smoothness_loss
        + consistency_weight * dense_loss
    )

    return loss

class SE3Loss(nn.Module):
    def forward(self, R_pred, R_gt, t_pred, t_gt):
        t_loss = ((t_pred - t_gt) ** 2).sum(dim=-1).mean()
        I = torch.eye(3).repeat(len(R_pred), 1, 1).to(R_pred.device)
        R_loss = (R_pred.transpose(1, 2) @ R_gt - I).norm(dim=(1, 2)).mean()
        loss = t_loss + R_loss

        return loss, R_loss, t_loss


class SE3LossTheirs(nn.Module):
    def forward(self, R_pred, R_gt, t_pred, t_gt):
        batch_size = len(t_gt)
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)

        R_loss = F.mse_loss(torch.matmul(R_pred.transpose(2, 1), R_gt), identity)
        t_loss = F.mse_loss(t_pred, t_gt)

        loss = R_loss + t_loss
        return loss, R_loss, t_loss