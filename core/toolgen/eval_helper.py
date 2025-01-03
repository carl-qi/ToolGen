from core.diffskill.visualization_utils import visualize_traj_actions
from core.toolgen.sample_rollout import sample_rollout_bc, sample_rollout_buffer
from core.diffskill.utils import visualize_trajs, aggregate_traj_info
from core.diffskill.hardcoded_eval_trajs import get_eval_traj
from chester import logger
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from tqdm import tqdm
import os
from core.diffskill.env_spec import get_threshold
from core.diffskill.utils import dict_add_prefix





def eval_traj(args, env, agent, epoch, tids=None, buffer=None):
    """ Run each skill on evaluation configurations; Save videos;
    Return raw trajs indexed by tid, time_step; Return aggregated info"""
    trajs, skill_info = [], {}
    if args.mask_gt_tool and args.env_name == 'MultiTool-v1':
        args.exclude_eval_idxes = range(600, 800)
    init_vs, target_vs = get_eval_traj(args.cached_state_path, exclude_idxes=args.exclude_eval_idxes, eval_subset=args.eval_subset, mode=args.eval_traj_mode)
    traj_counter = 0
    for init_v, target_v in tqdm(zip(init_vs, target_vs), desc="eval skills"):
        reset_key = {'init_v': init_v, 'target_v': target_v}
        if args.action_replay:
            assert buffer is not None
            traj = sample_rollout_buffer(env, args, buffer, reset_key)
        else:
            traj = sample_rollout_bc(env, agent, reset_key, buffer=buffer, cur_traj_idx=traj_counter)
        traj_counter += 1

        trajs.append(traj)
    # visualize_traj_actions(trajs, save_name=osp.join(logger.get_dir(), f"eval_actions_{epoch}.png"))

    keys = ['info_normalized_performance']
    fig, axes = plt.subplots(1, len(keys), figsize=(len(keys) * 5, 5))
    if len(keys) == 1:
        axes = [axes]
    for key_id, key in enumerate(keys):
        # Visualize traj
        if key == 'info_normalized_performance':
            visualize_trajs(trajs, key=key, ncol=10, save_name=osp.join(logger.get_dir(), f"eval_emd_epoch_{epoch}.gif"),
                            overlay_target=False, vis_target=True)
        # Visualize stats
        for traj_id, traj in enumerate(trajs):
            vals = traj[key]
            axes[key_id].plot(range(len(vals)), vals, label=f'traj_{traj_id}')
        axes[key_id].set_title(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(logger.get_dir(), f"eval_rollout_stats_epoch_{epoch}.png"))

    if args.opt_reset_pose:
        for traj_id, traj in enumerate(trajs):
            plot = traj['html_opt_plots']
            plot.write_html(osp.join(logger.get_dir(), f"opt_plots_traj_{traj_id}.html"))

    if args.visualize_flow:
        for traj_id, traj in enumerate(trajs):
            plots = traj['html_flow_plots']
            os.makedirs(osp.join(logger.get_dir(), f"eval_flow_plots_{epoch}_{traj_id}"), exist_ok=True)
            for i, plot in enumerate(plots):
                plot.write_html(osp.join(logger.get_dir(), f"eval_flow_plots_{epoch}_{traj_id}", f"step_{i}.html"))

    info = aggregate_traj_info(trajs)
    skill_info.update(**dict_add_prefix(info, f'eval/rollout/'))

    for traj in trajs:  # save space
        del traj['obses']
        del traj['states']
    return trajs, skill_info