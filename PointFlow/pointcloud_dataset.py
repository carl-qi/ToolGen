import os
from torch.utils.data import Dataset
import glob
import lzma
import pickle
import numpy as np
import torch
from tqdm import tqdm
from natsort import natsorted
from core.toolgen.vat_buffer import VatMartDataset
import random
import pytorch3d.transforms as transforms

from core.utils.utils import set_random_seed
from core.utils.pc_utils import decompose_pc
from core.toolgen.sample_rollout import get_correct_tid



class PointCloudDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.split = args.split
        self.data_subsample_n = 5
        self.data_train_ratio = 0.8
        self.data_dirs = natsorted(args.data_dirs)
        self.init_paths, self.target_paths = [], []
        self.sample_size = args.sample_size
        self.buffer = None
        self.all_pc, self.stat = None, None
        self.debug = args.debug
        self.tool_pcl_paths = natsorted(glob.glob(os.path.join(os.getcwd(), args.pcl_dir_path, '*.npy')))
        self.tool_particles = [torch.FloatTensor(np.load(path).reshape(1, 1000, 3)) for path in self.tool_pcl_paths]

        if args.load_from_buffer:
            assert self.sample_size == 1000, "Must use <= 1000 points when loading from buffer"
            self.load_buffer()
        else:
            self.load_init_target_paths()

        print("Total: {}, Train: {}, Validation: {}".format(self.N, len(self.train_idx), len(self.eval_idx)))
        print("Current split: {}".format(args.split))

        # Load actual data into self.all_pc
        if self.split == 'all':
            load_idx = self.all_idx
        elif self.split == 'train':
            load_idx = self.train_idx
        else:
            load_idx = self.eval_idx
        if 'load_set_from_buffer' in args.__dict__ and args.load_set_from_buffer:
            exit()
            self.load_set_dataset_from_buffer(load_idx)
        else:
            assert args.load_from_buffer
            self.load_dataset_from_buffer(load_idx)

        # Compute statistics
        self.compute_stats()

    def __len__(self):
        if self.split == 'all':
            return len(self.all_idx)
        elif self.split == 'train':
            return len(self.train_idx)
        else:
            return len(self.eval_idx)

    def load_dataset_from_buffer(self, idxes):
        batch_idxes = self.buffer.horizon * idxes
        pcs, _ = self.buffer.get_state(batch_idxes)
        target_v = self.buffer.buffer['target_v'][batch_idxes]
        self.target_vs = target_v
        goal_pcs = self.buffer.np_target_pc[target_v]
        self.all_pc = np.concatenate([pcs, goal_pcs], axis=1)


    def load_buffer(self):
        self.buffer = VatMartDataset(self.args)
        for data_dir in self.data_dirs:
            self.buffer.load(data_dir)
        if self.args.filter_buffer:
            self.buffer.generate_train_eval_split(train_ratio=self.data_train_ratio, filter=True, cached_state_path=self.args.cached_state_path)
        else:
            self.buffer.generate_train_eval_split(train_ratio=self.data_train_ratio, filter=False)
        self.N = self.buffer.new_num_traj
        self.train_idx = self.buffer.train_traj_idx
        self.eval_idx = self.buffer.eval_traj_idx[:100]

        # Load target
        from core.diffskill.utils import load_target_info
        target_info = load_target_info(self.args, 'cuda', load_set=False)
        self.buffer.__dict__.update(**target_info)

    # def load_init_target_paths(self):
    #     for data_dir in self.data_dirs:
    #         print("Adding path from: {}".format(data_dir))
    #         self.init_paths.extend(natsorted(glob.glob(os.path.join(data_dir, 'init/*[0-9].xz')))[::self.data_subsample_n])
    #         self.target_paths.extend(natsorted(glob.glob(os.path.join(data_dir, 'target/*[0-9].npy')))[::self.data_subsample_n])
    #     if self.debug:
    #         self.init_paths = self.init_paths[::10]
    #         self.target_paths = self.target_paths[::10]
    #     self.N = len(self.init_paths) + len(self.target_paths)

    # def load_pc(self, idx):
    #     if idx < len(self.init_paths):
    #         state_path = self.init_paths[idx]
    #         with lzma.open(state_path, 'rb') as f:
    #             state = pickle.load(f)
    #             x = state['state'][0]
    #     else:
    #         state_path = self.target_paths[idx - len(self.init_paths)]
    #         x = np.load(state_path, allow_pickle=True)
    #     return x

    # def load_dataset_from_buffer(self, idxes):
    #     self.all_pc = np.empty(shape=(len(idxes), self.sample_size, 3), dtype=np.float32)
    #     # For efficiency, just using the first `self.sample_size` number of points instead of calling `np.random.choice``
    #     pcs = self.buffer.buffer['states'][idxes, :3000].reshape(-1, 1000, 3)
    #     self.all_pc = pcs[:, :self.sample_size, :]
    #     del self.buffer

    # def load_set_dataset_from_buffer(self, idxes):
    #     self.all_pc = []
    #     # For efficiency, just using the first `self.sample_size` number of points instead of calling `np.random.choice``
    #     pcs = self.buffer.buffer['states'][idxes, :3000].reshape(-1, 1000, 3)
    #     labels = self.buffer.buffer['dbscan_labels'][idxes]
    #     for pc, label in tqdm(zip(pcs, labels), desc='loading set pointcloud'):
    #         set_pc = np.array(decompose_pc(pc, label, N=self.sample_size))
    #         self.all_pc.append(set_pc)
    #     goal_pcs = self.buffer.np_target_pc[:, :self.sample_size, :]
    #     goal_labels = self.buffer.np_target_dbscan
    #     for goal_pc, goal_label in tqdm(zip(goal_pcs, goal_labels), desc='loading goals set pointcloud'):
    #         goal_set_pc = np.array(decompose_pc(goal_pc, goal_label, N=self.sample_size))
    #         self.all_pc.append(goal_set_pc)

    #     self.all_pc = np.vstack(self.all_pc)
    #     print('Set dataset loaded. all_pc shape:', self.all_pc.shape)
    #     del self.buffer

    def compute_stats(self):
        assert self.all_pc is not None
        mean = np.mean(self.all_pc, axis=1)
        std = np.std((self.all_pc - np.mean(self.all_pc, axis=1, keepdims=True)).flatten())
        bb_min, bb_max = np.min(self.all_pc, axis=1, keepdims=True), np.max(self.all_pc, axis=1, keepdims=True)
        total_size = np.max(bb_max - bb_min, axis=-1, keepdims=True)
        padding = 0.1  # 'Padding applied to the sides (in total).'
        scale = total_size / (1 - padding)
        self.stat = {
            'mean': mean,
            'std': std,
            'bb_min': bb_min,
            'bb_max': bb_max,
            'scale': scale}

    def __getitem__(self, idx):
        raise NotImplementedError


class PointFlowDataset(PointCloudDataset):
    def __init__(self, args):
        super(PointFlowDataset, self).__init__(args)
        self.compute_stats()
        if self.split == 'val':
            self.stat['std'] = args.train_std
        self.normalize_pc()
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

    def normalize_pc(self):
        self.all_pc = self.all_pc - np.mean(self.all_pc, axis=1, keepdims=True)
        self.all_pc /= self.stat['std']

    def __getitem__(self, idx):
        pc = torch.FloatTensor(self.all_pc[idx])
        pcl_mean = torch.FloatTensor(self.stat['mean'][idx]).view(1, 3)
        m = self.all_pc[idx].mean(axis=0)
        tid = get_correct_tid(self.args.env_name, self.target_vs[idx], None)
        tool_pc = self.tool_particles[tid][0]
        tool_pc = tool_pc - torch.mean(tool_pc, dim=0, keepdim=True)

        # trajs
        if self.args.output_steps == 10:
            assert False
            action_idxes = (np.array([idx * self.buffer.horizon]) + np.arange(0, self.buffer.horizon, 5).reshape(1, -1)).flatten()
        else:
            if self.split == 'train':
                buffer_traj_idx = self.train_idx[idx]
            else:
                buffer_traj_idx = self.eval_idx[idx]
            action_idxes = np.array([buffer_traj_idx * self.buffer.horizon])
        _, pos_tool_trajs = self.buffer.get_state(action_idxes)
        pos_tool_trajs = torch.FloatTensor(pos_tool_trajs.reshape(len(action_idxes), -1, self.args.dimtool)[:, tid, :])  # use the current tool, resulting shape (horizon, dimtool)
        tool_pc_traj = transforms.quaternion_apply(pos_tool_trajs[:, 3:7].unsqueeze(1), tool_pc.unsqueeze(0)) + pos_tool_trajs[:, :3].unsqueeze(1)  # (horizon, N, 3)
        tool_pc_traj = tool_pc_traj.view(-1, 3)
        tool_pc_traj = tool_pc_traj - pcl_mean
        tool_pc_traj /= self.stat['std']

        tool_pc /= self.stat['std']
        train_points = torch.cat([pc, tool_pc], dim=0)
        # breakpoint()
        return {
            'idx': idx,
            'train_points': train_points,
            'train_label': tool_pc_traj,
            'mean': m.reshape(1, -1), 'std': self.stat['std'].reshape(1, -1), 'cate_idx': 0, 'sid': 0, 'mid': 0}