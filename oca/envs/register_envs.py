"""
oca/envs/register_envs.py \n
Registering custom environments with Gymnasium
"""

from gymnasium.envs.registration import register

register(
    id="FourRooms-v0",
    entry_point="oca.envs.fourrooms:FourRoomsEnv",
    max_episode_steps=1000,
    reward_threshold=1.0,
    kwargs={
        "render_args": {
            "show_goal": True,
            "force_update": True,
            "pause_delay": 1e-3,
        }
    },
)
