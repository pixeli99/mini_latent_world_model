from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy

env=MetaDriveEnv(dict(map="O",
                      agent_policy=ExpertPolicy,
                      log_level=50,
                      num_scenarios=1000,
                      start_seed=2,
                      traffic_density=0.2))
try:
    # run several episodes
    env.reset()
    for step in range(500):
        # simulation
        obs,_,_,_,info = env.step([0, 3])
        env.render(mode="topdown", 
                   window=False,
                   screen_record=True,
                   film_size=(500, 500),
                   screen_size=(128, 128),
                   scaling=None,
                   draw_contour=False,
                   num_stack=0,
                   target_agent_heading_up=True
                  )
        if info["arrive_dest"]:
            break
    env.top_down_renderer.generate_gif()
finally:
    env.close()