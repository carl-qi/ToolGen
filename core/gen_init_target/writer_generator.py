import numpy as np
from core.gen_init_target.state_generator import StateGenerator
import copy
from scipy.spatial.transform import Rotation as R

def rand(a, b):
    return np.random.random() * (b - a) + a

class WriterGenerator(StateGenerator):
    def __init__(self, *args, **kwargs):
        super(WriterGenerator, self).__init__(*args, **kwargs)
        self.env.reset()
        for i in range(20):
            self.env.step(np.zeros((12,)))
        self.initial_state = self.env.get_state()
        self.N = 100
        self.init_tool_states = []
        self.contact_points = []

    def write_pen_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        if save_init:
            self.save_init(i, state)

        pen_state = state['state'][-2]
        pen_state[0] = rand(0.45, 0.55)
        pen_state[1] = 0.17
        pen_state[2] = rand(0.45, 0.55)
        self.init_tool_states.append(pen_state)        
        self.taichi_env.set_state(**state)

        direction = rand(0, 2*np.pi)
        direction2 = rand(0, np.pi/2)
        for _ in range(10):
            self.env.step([np.cos(direction), -0.5, np.sin(direction) , 0, 0, 0]+[0,0,0,0,0,0])
        for _ in range(10):
            self.env.step([np.cos(direction+direction2), -0.5, np.sin(direction+direction2) , 0, 0, 0]+[0,0,0,0,0,0])
        state = self.env.get_state()
        self.save_target(i, state)

    def roll_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        if save_init:
            self.save_init(i, state)
        roller_state = state['state'][-2]
        coin = rand(0, 1)
        if coin < 0.25:
            roller_state[0] = rand(0.5, 0.52)
            roller_state[1] = 0.18
            roller_state[2] = rand(0.5, 0.52)
        elif coin < 0.5:
            roller_state[0] = rand(0.5, 0.52)
            roller_state[1] = 0.18
            roller_state[2] = rand(0.48, 0.5)
        elif coin < 0.75:
            roller_state[0] = rand(0.48, 0.5)
            roller_state[1] = 0.18
            roller_state[2] = rand(0.4, 0.48)
        else:
            roller_state[0] = rand(0.48, 0.5)
            roller_state[1] = 0.18
            roller_state[2] = rand(0.5, 0.52)
        direction = rand(0, np.pi/4)
        curr_rot = R.from_quat([0.707, 0, 0, 0.707])
        delta_rot = R.from_euler('xyz', angles=[0, direction, 0])
        result_quat = (curr_rot * delta_rot).as_quat()
        roller_state[4:7] = result_quat[:3]
        roller_state[3] = result_quat[-1]
        self.init_tool_states.append(roller_state)
        self.taichi_env.set_state(**state)
        
        contact_point = self.env.taichi_env.primitives[0].get_contact_point(0, state['state'][0][:1000].reshape(1000, 3)).reshape(1, 3)
        # print(contact_point)
        self.contact_points.append(contact_point)

        sign = 1. if coin < 0.5 else -1.
        for _ in range(10):
            self.env.step([sign * np.cos(direction), -0.5, sign * np.sin(direction) , 0, 0, 0] + [0,0,0,0,0,0])
        for _ in range(20):
            self.env.step([-sign * np.cos(direction), 0., -sign * np.sin(direction) , 0, 0, 0] + [0,0,0,0,0,0])   
        # for _ in range(10):
        #     self.env.step([-2 * np.cos(direction), 0, -2 * np.sin(direction) , 0, 0, 0] + [0,0,0,0,0,0])
        state = self.env.get_state()
        self.save_target(i, state)

    def write_knife_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        if save_init:
            self.save_init(i, state)
        pen_state = state['state'][-1]
        pen_state[0] = rand(0.48, 0.52)
        pen_state[1] = 0.2
        pen_state[2] = rand(0.48, 0.52)
        direction = rand(0, 2*np.pi)
        direction2 = rand(0, np.pi/4)
        curr_rot = R.from_quat([pen_state[4], pen_state[5], pen_state[6], pen_state[3]])
        delta_rot = R.from_euler('xyz', angles=[0, direction, 0])
        result_quat = (curr_rot * delta_rot).as_quat()
        pen_state[4:7] = result_quat[:3]
        pen_state[3] = result_quat[-1]
        self.init_tool_states.append(pen_state)
        self.taichi_env.set_state(**state)

        contact_point = self.env.taichi_env.primitives[1].get_contact_point(0, state['state'][0][:1000].reshape(1000, 3)).reshape(1, 3)
        # print(contact_point)
        self.contact_points.append(contact_point)

        for _ in range(10):
            self.env.step([0,0,0,0,0,0]+[0, -1.0, 0, 0, 0, 0])
        for _ in range(10):
            self.env.step([0,0,0,0,0,0]+[np.cos(direction), 0, np.sin(direction) , 0, 0, 0])
        for _ in range(10):
            self.env.step([0,0,0,0,0,0]+[np.cos(direction+direction2), 0, np.sin(direction+direction2) , 0, 0, 0])
        state = self.env.get_state()
        self.save_target(i, state)


    def _generate(self):
        # in-distribution cases
        index = 0
        # for i in range(self.N):
        #    self.write_pen_case(i, self.initial_state, save_init=True)
        for i in range(self.N):
           self.roll_case(i, self.initial_state, save_init=True)
        index += self.N
        for i in range(self.N):
           self.write_knife_case(i+index, self.initial_state, save_init=True)

        import os 
        import pickle
        with open(os.path.join(self.init_dir, 'init_tool_states.pkl'), 'wb') as f:
            pickle.dump(self.init_tool_states, f, protocol=4)
        with open(os.path.join(self.init_dir, 'contact_points.pkl'), 'wb') as f:
            pickle.dump(np.vstack(self.contact_points), f, protocol=4)
