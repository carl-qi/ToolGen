import random
import numpy as np
import torch
import argparse


def get_args(cmd=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument('--env_name', type=str, default='LiftSpread-v1')
    parser.add_argument('--num_env', type=int, default=1)  # Number of parallel environment
    parser.add_argument('--dataset_name', type=str, default='tmp')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--profiling", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--train_model", type=str, default='bc', choices=['bc', 'vatmart', 'taxpose', 'implicit'])

    # Env
    parser.add_argument("--dimtool", type=int, default=8)  # Dimension for representing state of the tool. 8 to incorporate the parallel gripper
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--env_loss_type", type=str, default='chamfer', choices=['emd', 'chamfer'])
    parser.add_argument("--adam_loss_type", type=str, default='chamfer', choices=['emd', 'chamfer'])
    parser.add_argument('--mask_gt_tool', type=bool, default=False)
    

    # Architecture
    parser.add_argument('--taxpose_embed_pose', type=bool, default=False)
    parser.add_argument('--taxpose_embed_goal', type=bool, default=True)
    parser.add_argument("--taxpose_arch", type=str, default='brianchuer-gc', choices=['brianchuer', 'brianchuer-gc'])
    parser.add_argument("--taxpose_embedding_dim", type=int, default=512)
    parser.add_argument("--input_mode", type=str, default='pc', choices=['pc'])
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--traj_feat_dim", type=int, default=128)
    parser.add_argument("--traj_z_dim", type=int, default=128)
    parser.add_argument("--task_feat_dim", type=int, default=32)
    parser.add_argument("--cp_feat_dim", type=int, default=32)
    parser.add_argument("--tool_feat_dim", type=int, default=256)
    parser.add_argument("--img_mode", type=str, default='rgbd')
    
    # Training
    parser.add_argument("--actor_type", type=str, default="v0")
    parser.add_argument("--value_fn_type", type=str, default="separate")
    parser.add_argument("--implicit_n_poses", type=int, default=50)
    parser.add_argument("--taxpose_n_poses", type=int, default=8)
    parser.add_argument("--use_bc_loss", type=bool, default=True)
    parser.add_argument('--use_contact', type=bool, default=False)
    parser.add_argument("--il_num_epoch", type=int, default=500)
    parser.add_argument("--taxpose_lr", type=float, default=1e-3)
    parser.add_argument("--vat_lr", type=float, default=1e-3)
    parser.add_argument("--il_lr", type=float, default=1e-3)
    parser.add_argument("--il_eval_freq", type=int, default=5)
    parser.add_argument("--lbd_kl", type=float, default=0.)
    parser.add_argument("--lbd_recon", type=float, default=0.)
    parser.add_argument("--lbd_dir", type=float, default=0.)
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--step_per_epoch", type=int, default=5000000)
    parser.add_argument("--step_warmup", type=int, default=2000)
    parser.add_argument("--hindsight_goal_ratio", type=float, default=0.5)
    parser.add_argument("--obs_noise", type=float, default=0.005)  # Noise for the point cloud in the original space
    parser.add_argument("--num_tools", type=int, default=2)
    parser.add_argument('--train_tool_idxes', type=list, default=[0])
    parser.add_argument("--train_modules", type=list, default=['policy', 'value_fn'])  # Modules to train
    parser.add_argument("--load_modules", type=list, default=['policy', 'value_fn'])  # Modules to load
    parser.add_argument('--filter_buffer', type=bool, default=False)
    parser.add_argument('--random_init_pose', type=bool, default=False)

    # Evaluation
    # parser.add_argument("--test_model", type=str, default='bc', choices=['bc', 'vatmart', 'taxpose-bc'])
    parser.add_argument("--bc_resume_path", default=None)
    parser.add_argument('--eval_traj_mode', type=str, default='all')
    parser.add_argument("--pointflow_resume_path", default=None)
    parser.add_argument("--taxpose_resume_path", default=None)
    parser.add_argument("--use_gt_tool_and_reset", type=bool, default=False)
    parser.add_argument("--use_gt_tool", type=bool, default=False)
    parser.add_argument("--fit_training_tool", type=bool, default=False)
    parser.add_argument("--action_replay", type=bool, default=False)
    parser.add_argument("--exclude_eval_idxes", type=list, default=[])
    parser.add_argument("-eval_subset", type=bool, default=False)
    parser.add_argument("--visualize_flow", type=bool, default=False)
    parser.add_argument("--opt_reset_pose", type=bool, default=False)
    parser.add_argument("--opt_traj_deltas", type=bool, default=False)

    if cmd:
        args = parser.parse_args()
    else:
        args = parser.parse_args("")

    return args
