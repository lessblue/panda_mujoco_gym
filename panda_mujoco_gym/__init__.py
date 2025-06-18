import os
from gymnasium.envs.registration import register

ENV_IDS = []

for task in ["Slide", "Push", "PickAndPlace"]:
    for reward_type in ["sparse", "dense"]:
        reward_suffix = "Dense" if reward_type == "dense" else "Sparse"
        env_id = f"Franka{task}{reward_suffix}-v0"

        register(
            id=env_id,
            entry_point=f"panda_mujoco_gym.envs:Franka{task}Env",
            kwargs={"reward_type": reward_type},
            max_episode_steps=50,
        )

        ENV_IDS.append(env_id)
        
STACK_TASK_MAX_STEPS = 150  # 为Stack任务定义一个更合理的步长

for reward_type in ["sparse", "dense"]:
    reward_suffix = "Dense" if reward_type == "dense" else "Sparse"
    env_id = f"FrankaStack{reward_suffix}-v0"

    register(
        id=env_id,
        entry_point=f"panda_mujoco_gym.envs:FrankaStackEnv",
        kwargs={"reward_type": reward_type},
        max_episode_steps=STACK_TASK_MAX_STEPS,  # <<< 为Stack定义的150步
    )

    ENV_IDS.append(env_id)
