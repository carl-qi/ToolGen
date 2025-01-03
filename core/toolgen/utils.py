import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import pytorch3d.transforms as transforms
from core.utils.pcl_visualization_utils import create_flow_plot, create_pcl_plot

def optimize_delta_poses(all_target_pcls, init_tool_pcl, delta_poses, best_reset_pose):
    optimizer = Adam([delta_poses], lr=1e-2)
    pbar = tqdm(range(100))
    for char in pbar:
        # all_opt_poses = torch.zeros((len(all_target_pcls)+1, 1, 7)).cuda()
        all_opt_poses = [torch.zeros((1, 7)).cuda() for _ in range(len(all_target_pcls)+1)]
        # all_opt_poses[0, :, :] = best_reset_pose
        all_opt_poses[0] = best_reset_pose
        all_opt_pcls = []
        for t in range(len(all_target_pcls)):
            cur_dir = transforms.quaternion_multiply(
                transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(delta_poses[t, 3:], "XYZ")), 
                # all_opt_poses[t, 0, 3:].clone()
                all_opt_poses[t][0, 3:]
                ).view(1, 1, 4)
            # cur_loc = (all_opt_poses[t, 0, :3].clone() + delta_poses[t, :3]).view(1, 1, 3)
            cur_loc = (all_opt_poses[t][0, :3] + delta_poses[t, :3]).view(1, 1, 3)
            # all_opt_poses[t+1:t+2, :, :] = torch.cat([cur_loc, cur_dir], dim=-1)
            all_opt_poses[t+1][:, :] = torch.cat([cur_loc, cur_dir], dim=-1).view(1, 7)

        all_opt_poses = torch.vstack(all_opt_poses).view(len(all_target_pcls)+1, 1, 7)
        all_opt_pcls = transforms.quaternion_apply(all_opt_poses[1:, :, 3:7], init_tool_pcl) + all_opt_poses[1:, :, :3]
        cur_loss, _ = chamfer_distance(torch.vstack(all_target_pcls).view(len(all_target_pcls), -1, 3), all_opt_pcls, batch_reduction='mean', one_way=True)
        cur_loss += 0.1 * torch.norm(delta_poses, dim=-1).mean()
        # loss += cur_loss
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        pbar.set_description("Step {}, Loss: {}".format(char, cur_loss.item()))
    
    all_poses = all_opt_poses[1:, :, :].detach()
    all_opt_pcls = transforms.quaternion_apply(all_poses[:, :, 3:7], init_tool_pcl) + all_poses[:, :, :3]
    return delta_poses.detach(), all_opt_pcls.detach(), all_poses.detach()

def optimize_reset_pose(scene_pcls, init_pcls, target_pcls, init_poses, init_locs):
    optimizer = Adam([init_poses, init_locs], lr=1e-3)
    pbar = tqdm(range(1000))
    B = init_poses.shape[0]
    all_opt_poses = []
    dough_points = scene_pcls[:, :1000, :3].tile(B, 1, 1)
    for char in pbar:
        proj_poses = init_poses / torch.norm(init_poses, dim=-1).view(B, 1, -1)
        cur_pcls = transforms.quaternion_apply(proj_poses, init_pcls) + init_locs
        all_opt_poses.append(torch.cat([init_locs, proj_poses], dim=-1).detach())
        loss, _ = chamfer_distance(target_pcls, cur_pcls, batch_reduction=None, one_way=True)
        loss2, _ = chamfer_distance(dough_points, cur_pcls, batch_reduction=None, one_way=True)
        all_loss = loss - 0.1 * loss2
        l = torch.mean(all_loss, dim=0)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
        pbar.set_description("Step {}, Loss: {}".format(char, l.item()))
    best_idx = torch.argmin(all_loss, dim=0)
    all_opt_poses = torch.vstack(all_opt_poses).view(-1, B, 7)
    final_opt_pose = all_opt_poses[-1, best_idx, :].view(1, 7)
    init_pcl = init_pcls[0].view(1, -1, 3)
    final_pcl = transforms.quaternion_apply(final_opt_pose[:, 3:], init_pcl) + final_opt_pose[:, :3]
    plot = create_pcl_plot(scene_pcls[0], target_pcls[0], init_pcl[0], final_pcl[0])
    return final_opt_pose, final_pcl, plot

# input sz bszx3x2
def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

# batch geodesic loss for rotation matrices
def bgdR(Rgts, Rps):
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    return torch.acos(theta)

# 6D-Rot loss
# input sz bszx6
def get_6d_rot_loss(pred_6d, gt_6d):
    pred_Rs = bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    gt_Rs = bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    theta = bgdR(gt_Rs, pred_Rs)
    return theta

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


# batch geodesic loss for rotation matrices
def bgdR(Rgts, Rps):
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    return torch.acos(theta) / 2

def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
    sqrt: bool = True,
    one_way: bool = False,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    if sqrt:
        cham_x = cham_x.sqrt()
        cham_y = cham_y.sqrt()

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        x_lengths_clamped = x_lengths.clamp(min=1)
        y_lengths_clamped = y_lengths.clamp(min=1)
        cham_x /= x_lengths_clamped
        cham_y /= y_lengths_clamped
        if return_normals:
            cham_norm_x /= x_lengths_clamped
            cham_norm_y /= y_lengths_clamped

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else max(N, 1)
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    if one_way:
        cham_dist = cham_x
    else:
        cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals