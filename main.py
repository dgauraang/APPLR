#import rospy
import gym
import os
#import jackal_envs
print('Successfully imported rospy')


os.system('export DISPLAY=:1.0')
steps = 100
eps = 10
env = gym.make('jackal_navigation-v0', world_name = 'sequential_applr_testbed', gui = 'false', VLP16 = 'false', init_position = [45, 1, 0], goal_position = [-4.2, .16, 0])

with open('/buffer/log', 'w') as log:
    for ep in range(eps):
        env.reset()
        d = False
        step = 0
        log.write('Episode: {}'.format(episode))
        while not d and step < steps:
            o, r, d, _ = env.step()
            step += 1
            log.write('Step: {}'.format(step))

env.close()
