import gym
import rospy
import rospkg
import roslaunch
import time
import numpy as np
import os
import subprocess
import jackal_envs

from gym import utils, spaces
from std_srvs.srv import Empty
import actionlib
from gym.utils import seeding

from .gazebo_simulation import GazeboSimulation
from .navigation_stack import  NavigationStack

range_dict = {
    'max_vel_x': [0.1, 2],
    'max_vel_theta': [0.314, 3.14],
    'vx_samples': [4, 12],
    'vtheta_samples': [8, 40],
    'path_distance_bias': [0.1, 1.5],
    'goal_distance_bias': [0.1, 2]
}

action_lows = [0.1, 0.314, 4, 8, 0.1, 0.1]
action_highs = [2, 3.14, 12, 40, 1.5, 2]

class GazeboJackalNavigationEnv(gym.Env):

    def __init__(self, world_name = 'sequential_applr_testbed', VLP16 = 'true', gui = 'false', camera = 'false',
                init_position = [45, 1, 0], goal_position = [45, -1, 0], max_step = 20, time_step = 1,
                param_init = [0.5, 1.57, 6, 20, 0.75, 1],
                param_list = ['max_vel_x', 'max_vel_theta', 'vx_samples', 'vtheta_samples', 'path_distance_bias', 'goal_distance_bias']):
        gym.Env.__init__(self)

        os.system("killall -9 rosmaster")
        os.system("killall -9 gzclient")
        os.system("killall -9 gzserver")
        os.system("killall -9 roscore")
        os.system('hostname -I')
        self.world_name = world_name
        self.VLP16 = True if VLP16=='true' else False
        self.gui = True if gui=='true' else False
        self.max_step = max_step
        self.time_step = time_step
        self.goal_position = goal_position
        self.param_init = param_init
        self.param_list = param_list

        # Launch gazebo and navigation demo
        # Should have the system enviroment source to jackal_helper
        rospack = rospkg.RosPack()
        #BASE_PATH = '/'
        BASE_PATH = '/jackal_ws/src/jackal_simulator/jackal_gazebo'
        NAV_PATH = '/jackal_ws/src/jackal/jackal_navigation'
        gui = 'false' 
        
        self.gazebo_process = subprocess.Popen(['roslaunch', \
                                                os.path.join(BASE_PATH, 'launch', world_name + '.launch'),
                                                'gui:=' + gui,
                                                'config:=' + 'front_laser' 
                                                ], creationflags=subprocess.DETACHED_PROCESS)

        self.odom_process = subprocess.Popen(['roslaunch', \
                                               os.path.join(NAV_PATH, 'launch', 'odom_navigation_demo.launch'),])
        print('Done launching gazebo!')
        print('Done launching gazebo!')
        print('Done launching gazebo!')
        rospy.set_param('/use_sim_time', True)
        rospy.init_node('gym', anonymous=True)

        self.gazebo_sim = GazeboSimulation(init_position = init_position)
        self.navi_stack = NavigationStack(goal_position = goal_position)
        
        # Dynamically set parameter bounds based on the parameters the model will learn 
        low = np.array([range_dict[param_list[i]][0] for i in range(len(param_list))])
        high = np.array([range_dict[param_list[i]][1] for i in range(len(param_list))])

        # The action space is continuous, with each action as the number of different 
        # parameters to be adjusted        
        self.action_space = spaces.Box(low=low, high=high, shape=(len(param_list),))
        self.reward_range = (-np.inf, np.inf)
        if VLP16 == 'true':
            self.observation_space = spaces.Box(low=np.array([-1]*(2095)), # a hard coding here
                                                high=np.array([1]*(2095)),
                                                dtype=np.float)
        elif VLP16 == 'false':
            self.observation_space = spaces.Box(low=np.array([-1]*(721+len(self.param_list))), # a hard coding here
                                                high=np.array([1]*(721+len(self.param_list))),
                                                dtype=np.float)

        self._seed()
        self.navi_stack.set_global_goal()
        print('Finished initialization')
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _observation_builder(self, laser_scan, local_goal):
        '''
        Observation is the laser scan plus local goal. Episode ends when the
        between global goal and robot positon is less than 0.4m. Reward is set
        to -1 for each step plus a 0.9 reward for decreasing distance between the previous position
        and the goal (which makes the net reward 0.1 instead of -1).

        A reward of -1000 is received if the robot flips over, and the environment restarts.       
        '''
        scan_ranges = np.array(laser_scan.ranges)
        scan_ranges[scan_ranges == np.inf] = 20
        local_goal_position = np.array([np.arctan((local_goal.position.x + .0000001)/(local_goal.position.y + .00000001))])
        params = []
        params_normal = []
        for pn in self.param_list:
            params.append(self.navi_stack.get_navi_param(pn))
            params_normal.append(params[-1]/float(range_dict[pn][1]))
        state = np.concatenate([scan_ranges/20.0, local_goal_position/np.pi, np.array(params_normal)])

        pr = np.array([self.navi_stack.robot_config.X, self.navi_stack.robot_config.Y])
        gpl = np.array(self.goal_position[:2])
        self.gp_len = np.sqrt(np.sum((pr-gpl)**2))
        if self.gp_len < 0.4 or pr[0] >= 50 or self.step_count >= self.max_step:
            done = True
            print('REACHED GOAL!!!')
        else:
            done = False

        return state, -1, done, {'params': params}
    
    # Maps each action from a range of [-1, 1] to the appropriate parameter bounds
    def map_action(self, action):
        low = self.action_space.low 
        high = self.action_space.high

        new_action = low + (action + 1)/2 * (high - low)
        return new_action 

    def step(self, action, sampled=False):
        # Check if the number of actions is equivalent to the number of parameters
        assert len(action) == self.action_space.shape[0]
        self.step_count += 1
        i = 0

        # Only wrap the action if it was sampled from an appropriate range
        action = self.map_action(action)
        
        action_vals = [] 
        # Set the parameters to values specified by the agent's action
        for param_value, param_name in zip(action, self.param_list):
            if 'samples' in param_name:
                param_value = np.rint(param_value)
            action_vals.append(param_value)
            self.navi_stack.set_navi_param(param_name, param_value.item())

        print('Action:', action_vals)
        # Sleep for 5s (a hyperparameter that can be tuned)
        rospy.sleep(self.time_step)

        # Collect the laser scan data
        laser_scan = self.gazebo_sim.get_laser_scan()
        local_goal = self.navi_stack.get_local_goal()

        return self._observation_builder(laser_scan, local_goal)

    def reset(self):

        self.step_count = 0
        # reset robot in odom frame clear_costmap
        self.navi_stack.reset_robot_in_odom()
        # Resets the state of the environment and returns an initial observation.
        self.gazebo_sim.reset()
        # reset max_vel_x value
        for init, pn in zip(self.param_init, self.param_list):
            self.navi_stack.set_navi_param(pn, init)

        #read laser data
        self.navi_stack.clear_costmap()
        rospy.sleep(0.1)
        self.navi_stack.clear_costmap()

        laser_scan = self.gazebo_sim.get_laser_scan()
        local_goal = self.navi_stack.get_local_goal()
        self.navi_stack.set_global_goal()

        state, _, _, _ = self._observation_builder(laser_scan, local_goal)

        return state

    def close(self):
        os.system("killall -9 rosmaster")
        os.system("killall -9 gzclient")
        os.system("killall -9 gzserver")
        os.system("killall -9 roscore")

if __name__ == '__main__':
    env = GazeboJackalNavigationEnv()
    env.reset()
    print(env.step(0))
    env.unpause()
    time.sleep(30)
    env.close()
