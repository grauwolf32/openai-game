from gym.envs.registration import register

register(
    id='pursuite-v0',
    entry_point='gym_pursuite.envs:PursuitGameEnv',
)
register(
    id='gathering-v0',
    entry_point='gym_pursuite.envs:GatheringGameEnv',
)
