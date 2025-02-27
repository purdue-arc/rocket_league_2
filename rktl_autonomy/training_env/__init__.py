from gymnasium.envs.registration import register

register(
    id="training_env/TouchBall-v0",
    entry_point="training_env.envs.grid_world:TouchBallNoPhysicsEnv",
    max_episode_steps=300,
    reward_threshold=None,
    kwargs = {"render_mode":"human"}
)