from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod

from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.utils import generate_gif
from IPython.display import Image, clear_output
import cv2

# map_config={BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM, 
#             BaseMap.GENERATE_CONFIG: 3,  # 3 block
#             BaseMap.LANE_WIDTH: 3.5,
#             BaseMap.LANE_NUM: 2}
# map_config["config"]=3

# env=MetaDriveEnv(dict(map_config=map_config,
#                       agent_policy=ExpertPolicy,
#                       log_level=50,
#                       num_scenarios=1000,
#                       start_seed=0,
#                       traffic_density=0.2))
nuscenes_data=AssetLoader.file_path("/18940970966/nuplan_mini", "meta_drive", unix_style=False) 
env = ScenarioEnv(
    {
        "reactive_traffic": False,
        "use_render": False,
        "agent_policy": ReplayEgoCarPolicy,
        "data_directory": nuscenes_data,
        "num_scenarios": 1800,
    }
)
try:
    # run several episodes
    env.reset(seed=222)

    policy = env.engine.get_policy(env.current_track_agent.name)
    for step in range(500):
        # simulation
        action = policy.get_action_info()
        print(action)
        obs,_,_,_,info = env.step([0, 3])
        env.render(mode="topdown", 
                   window=False,
                   screen_record=True,
                   film_size=(1600, 1600),
                   screen_size=(128, 128),
                   scaling=4,
                   draw_contour=False,
                   num_stack=0,
                   target_agent_heading_up=True
                  )
        if info["arrive_dest"]:
            break
    env.top_down_renderer.generate_gif()
finally:
    env.close()