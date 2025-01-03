import numpy as np
from core.gen_init_target.state_generator import StateGenerator
import copy
from scipy.spatial.transform import Rotation as R

def rand(a, b):
    return np.random.random() * (b - a) + a

def cut(state, cut_loc, slide_a, slide_b):
    ns = copy.deepcopy(state)
    flag = ns['state'][0][:, 0] <= cut_loc
    ns['state'][0][flag, 0] -= slide_a
    ns['state'][0][np.logical_not(flag), 0] += slide_b
    return ns, flag

class MultitoolGenerator(StateGenerator):
    def __init__(self, *args, **kwargs):
        super(MultitoolGenerator, self).__init__(*args, **kwargs)
        self.env.reset()
        self.num_tools = 5
        for i in range(20):
            self.env.step(np.zeros((6*self.num_tools,)))
        self.initial_state = self.env.get_state()
        self.N = 200
        self.init_tool_states = []
        self.contact_points = []
        self.traj_actions = {}

    def write_large_pen_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        pos = np.mean(state['state'][0], axis=0)
        size = np.array([0.3, 0.1, 0.3])
        pos[1] = size[1] / 2
        state['state'][0] = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * size) + pos
        if save_init:
            self.save_init(i, state)

        action_idxes = np.arange(2*6, 3*6)
        pen_state = state['state'][6]
        pen_state[0] = rand(0.48, 0.52)
        pen_state[1] = 0.16
        pen_state[2] = rand(0.48, 0.52)
        state['state'][6] = pen_state
        self.taichi_env.set_state(**state)

        self.init_tool_states.append(pen_state)        
        contact_point = self.env.taichi_env.primitives[2].get_contact_point(0, state['state'][0][:1000].reshape(1000, 3)).reshape(1, 3)
        self.contact_points.append(contact_point)
        direction = rand(0, np.pi*2)
        direction2 = rand(-np.pi/2, np.pi/2)
        actions = []
        for _ in range(5):
            action = np.zeros((6*self.num_tools,))
            action[action_idxes] = [0, -0.5, 0, 0, 0, 0]
            self.env.step(action)
            actions.append(action)
        for _ in range(8):
            action = np.zeros((6*self.num_tools,))
            action[action_idxes] = [np.cos(direction), 0, np.sin(direction) , 0, 0, 0]
            self.env.step(action)
            actions.append(action)
        for _ in range(8):
            action = np.zeros((6*self.num_tools,))
            action[action_idxes] = [np.cos(direction+direction2), 0, np.sin(direction+direction2) , 0, 0, 0]
            self.env.step(action)
            actions.append(action)
        state = self.env.get_state()
        self.save_target(i, state)
        self.traj_actions[i] = actions

    def write_small_pen_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        pos = np.mean(state['state'][0], axis=0)
        size = np.array([0.3, 0.1, 0.3])
        pos[1] = size[1] / 2
        state['state'][0] = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * size) + pos
        if save_init:
            self.save_init(i, state)
        # TODO MOVE THIS BACK
        action_idxes = np.arange(3*6, 4*6)
        pen_state = state['state'][7]
        pen_state[0] = rand(0.48, 0.52)
        pen_state[1] = 0.16
        pen_state[2] = rand(0.48, 0.52)
        state['state'][7] = pen_state
        self.taichi_env.set_state(**state)

        self.init_tool_states.append(pen_state)        
        contact_point = self.env.taichi_env.primitives[3].get_contact_point(0, state['state'][0][:1000].reshape(1000, 3)).reshape(1, 3)
        self.contact_points.append(contact_point)
        direction = rand(0, np.pi*2)
        direction2 = rand(-np.pi/2, np.pi/2)
        actions = []
        for _ in range(5):
            action = np.zeros((6*self.num_tools,))
            action[action_idxes] = [0, -0.5, 0, 0, 0, 0]
            self.env.step(action)
            actions.append(action)
        for _ in range(8):
            action = np.zeros((6*self.num_tools,))
            action[action_idxes] = [np.cos(direction), 0, np.sin(direction) , 0, 0, 0]
            self.env.step(action)
            actions.append(action)
        for _ in range(8):
            action = np.zeros((6*self.num_tools,))
            action[action_idxes] = [np.cos(direction+direction2), 0, np.sin(direction+direction2) , 0, 0, 0]
            self.env.step(action)
            actions.append(action)
        state = self.env.get_state()
        self.save_target(i, state)
        self.traj_actions[i] = actions

    def roll_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        pos = np.mean(state['state'][0], axis=0)
        size = np.array([0.2, 0.1, 0.2])
        pos[1] = size[1] / 2
        og_state = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * size) + pos
        state['state'][0] = og_state
        offset_x, offset_z = rand(-0.02, 0.02), rand(-0.02, 0.02)

        if save_init:
            self.save_init(i, state)

        # Translation
        roller_state = state['state'][6]
        roller_state[0] = pos[0] + offset_x
        roller_state[1] = 0.17
        roller_state[2] = pos[2] + offset_z
        curr_rot = R.from_quat([0.707, 0, 0, 0.707])  # type: ignore
        ang = rand(-np.pi/4, np.pi/4)
        delta_rot = R.from_euler('xyz', angles=[0, 0, ang])  # type: ignore
        result_quat = (curr_rot * delta_rot).as_quat()

        roller_state[4:7] = result_quat[:3]
        roller_state[3] = result_quat[-1]
        state['state'][6] = roller_state
        
        w = rand(0.25, 0.28)
        h = rand(0.02, 0.03)
        
        size = np.array([w, h, 0.2])
        # Init dough state
        dough_state = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * size)
        # Rotate
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        xz = np.hstack([dough_state[:, 0:1], dough_state[:, 2:3]]).T
        new_xz = rot@xz
        dough_state[:, 0] = new_xz[0]
        dough_state[:, 2] = new_xz[1]
        # Translate
        coin = rand(0, 1)
        sign = 1 if coin > 0.5 else -1
        dir = rot @ np.array([1, 0])
        dough_state += np.array([pos[0] + sign*dir[0]*0.1, h + h, pos[2] + sign*dir[1]*0.1])
        # Set dough state
        full_state = np.vstack([dough_state, og_state])
        dough_idxes = np.random.choice(len(full_state), size=len(state['state'][0]), replace=False)
        state['state'][0] = full_state[dough_idxes]
        self.init_tool_states.append(roller_state)
        self.taichi_env.set_state(**state)
        
        contact_point = self.env.taichi_env.primitives[2].get_contact_point(0, state['state'][0][:1000].reshape(1000, 3)).reshape(1, 3)
        # print(contact_point)
        self.contact_points.append(contact_point)
        state = self.env.get_state()
        self.save_target(i, state)

    def cut_knife_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        pos = np.mean(state['state'][0], axis=0)
        size = np.array([0.3, 0.1, 0.3])
        pos[1] = size[1] / 2
        state['state'][0] = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * size) + pos
        if save_init:
            self.save_init(i, state)

        knife_state = state['state'][8]
        coin = rand(0, 1)
        if coin < 0.25:
            knife_state[0] = rand(0.57, 0.60)
            knife_state[2] = rand(0.57, 0.60)
        elif coin < 0.5:
            knife_state[0] = rand(0.4, 0.43)
            knife_state[2] = rand(0.4, 0.43)
        elif coin < 0.75:
            knife_state[0] = rand(0.57, 0.60)
            knife_state[2] = rand(0.4, 0.43)
        else:
            knife_state[0] = rand(0.4, 0.43)
            knife_state[2] = rand(0.57, 0.60)
        knife_state[1] = 0.18

        direction = rand(np.pi/6, np.pi/4) * (1 if coin > 0.5 else -1)
        curr_rot = R.from_quat([knife_state[4], knife_state[5], knife_state[6], knife_state[3]])  # type: ignore
        delta_rot = R.from_euler('xyz', angles=[0, direction, 0])  # type: ignore
        result_quat = (curr_rot * delta_rot).as_quat()
        knife_state[4:7] = result_quat[:3]
        knife_state[3] = result_quat[-1]
        self.init_tool_states.append(knife_state)
        state['state'][8] = knife_state
        self.taichi_env.set_state(**state)
        contact_point = self.env.taichi_env.primitives[4].get_contact_point(0, state['state'][0][:1000].reshape(1000, 3)).reshape(1, 3)
        # print(contact_point)
        self.contact_points.append(contact_point)

        ns = copy.deepcopy(state)
        if coin < 0.25:
            flag = np.logical_and(ns['state'][0][:, 0] >= knife_state[0], ns['state'][0][:, 2] >= knife_state[2])
            ns['state'][0][flag, 0] += 0.1
            ns['state'][0][flag, 2] += 0.1
        elif coin < 0.5:
            flag = np.logical_and(ns['state'][0][:, 0] <= knife_state[0], ns['state'][0][:, 2] <= knife_state[2])
            ns['state'][0][flag, 0] -= 0.1
            ns['state'][0][flag, 2] -= 0.1
        elif coin < 0.75:
            flag = np.logical_and(ns['state'][0][:, 0] >= knife_state[0], ns['state'][0][:, 2] <= knife_state[2])
            ns['state'][0][flag, 0] += 0.1
            ns['state'][0][flag, 2] -= 0.1
        else:
            flag = np.logical_and(ns['state'][0][:, 0] <= knife_state[0], ns['state'][0][:, 2] >= knife_state[2])
            ns['state'][0][flag, 0] -= 0.1
            ns['state'][0][flag, 2] += 0.1

        self.taichi_env.set_state(**ns)
        state = self.env.get_state()
        self.save_target(i, state)

    def scoop_small_bowl_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        pos = np.mean(state['state'][0], axis=0)
        if save_init:
            self.save_init(i, state)

        action_idxes = np.arange(0*6, 1*6)
        bowl_state = state['state'][4]
        bowl_state[0] = rand(0.45, 0.55)
        bowl_state[1] = 0.18
        bowl_state[2] = rand(0.42, 0.52)
        state['state'][4] = bowl_state
        self.init_tool_states.append(bowl_state)    
        self.taichi_env.set_state(**state)    
        contact_point = self.env.taichi_env.primitives[0].get_contact_point(0, state['state'][0][:1000].reshape(1000, 3)).reshape(1, 3)
        self.contact_points.append(contact_point)
        actions = []
        for _ in range(5):
            action = np.zeros((6*self.num_tools,))
            action[action_idxes] = [0, -1, 0, 0, 0, 0]
            self.env.step(action)
            actions.append(action)
        for t in range(20):
            action = np.zeros((6*self.num_tools,))
            if t >= 15:
                action[action_idxes] = [0, 0, 0.5, -1, 0, 0]
            else:
                action[action_idxes] = [0, 0, 0, -1, 0, 0]
            self.env.step(action)
            actions.append(action)
        for _ in range(15):
            action = np.zeros((6*self.num_tools,))
            action[action_idxes] = [0, 1, 0, 0, 0, 0]
            self.env.step(action)
            actions.append(action)
        state = self.env.get_state()
        self.save_target(i, state)
        self.traj_actions[i] = actions 

    def scoop_large_bowl_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        pos = np.mean(state['state'][0], axis=0)
        if save_init:
            self.save_init(i, state)

        action_idxes = np.arange(1*6, 2*6)
        bowl_state = state['state'][5]
        bowl_state[0] = rand(0.45, 0.55)
        bowl_state[1] = 0.19
        bowl_state[2] = rand(0.42, 0.52)
        state['state'][5] = bowl_state

        self.init_tool_states.append(bowl_state)
        self.taichi_env.set_state(**state) 
        contact_point = self.env.taichi_env.primitives[1].get_contact_point(0, state['state'][0][:1000].reshape(1000, 3)).reshape(1, 3)
        self.contact_points.append(contact_point)
        actions = []
        for _ in range(8):
            action = np.zeros((6*self.num_tools,))
            action[action_idxes] = [0, -1, 0, 0, 0, 0]
            self.env.step(action)
            actions.append(action)
        for t in range(20):
            action = np.zeros((6*self.num_tools,))            
            if t >= 15:
                action[action_idxes] = [0, 0, 0.8, -1, 0, 0]
            else:
                action[action_idxes] = [0, 0, 0, -1, 0, 0]
            self.env.step(action)
            actions.append(action)
        for _ in range(20):
            action = np.zeros((6*self.num_tools,))
            action[action_idxes] = [0, 1, 0, 0, 0, 0]
            self.env.step(action)
            actions.append(action)
        state = self.env.get_state()
        self.save_target(i, state)
        self.traj_actions[i] = actions

    def _generate(self):
        # in-distribution cases
        index = 0
        for i in range(self.N):
           self.roll_case(i, self.initial_state, save_init=True)
        index += self.N
        for i in range(self.N):
           self.write_small_pen_case(i+index, self.initial_state, save_init=True)
        index += self.N
        for i in range(self.N):
           self.write_large_pen_case(i+index, self.initial_state, save_init=True)
        index += self.N
        for i in range(self.N):
           self.cut_knife_case(i+index, self.initial_state, save_init=True)
        index += self.N
        for i in range(self.N):
           self.scoop_small_bowl_case(i+index, self.initial_state, save_init=True)
        index += self.N
        for i in range(self.N):
           self.scoop_large_bowl_case(i+index, self.initial_state, save_init=True)
        index += self.N

        import os 
        import pickle
        with open(os.path.join(self.init_dir, 'init_tool_states.pkl'), 'wb') as f:
            pickle.dump(self.init_tool_states, f, protocol=4)
        with open(os.path.join(self.init_dir, 'contact_points.pkl'), 'wb') as f:
            pickle.dump(np.vstack(self.contact_points), f, protocol=4)
        with open(os.path.join(self.init_dir, 'traj_actions.pkl'), 'wb') as f:
            pickle.dump(self.traj_actions, f, protocol=4)
