from gym.envs.registration import register

register(
    id='jackal_navigation-v0',
    entry_point='jackal_envs.envs:GazeboJackalNavigationEnv',
)
