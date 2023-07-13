import numpy as np

from pyflyt_rail_env import Environment

env = Environment(render_mode="human")

obs, _ = env.reset()
action = np.zeros(*env.action_space.shape)

while True:

    # print(env.hidden_state["seg_img"].max())
    # print(env.hidden_state["seg_img"].min())
    # exit()
    track_position = env.track_state()

    action[0] = 3.0
    action[1] = track_position[0] * 3.0
    action[2] = track_position[1] * 3.0
    action[3] = 2.0 - env.drone.state[-1][-1]

    obs, rew, term, trunc, info = env.step(action)

    if term or trunc:
        env.close()
        exit()
