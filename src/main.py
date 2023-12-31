import cv2
import numpy as np

from pyflyt_rail_env import Environment

env = Environment(render_mode="human")
# env = Environment(render_mode=None)

obs, _ = env.reset()
term = False
trunc = False
action = np.zeros(*env.action_space.shape)

while not (term or trunc):
    track_position = env.track_state
    action[0] = env.target_velocity
    action[1] = track_position[0] * 1.0
    action[2] = track_position[1] * 1.0
    action[3] = env.target_height - env.drone.state[-1][-1]

    obs, rew, term, trunc, info = env.step(action)
    print(info)

    cv2.imshow("something", obs["seg_img"][..., 0].astype(np.uint8) * 255)
    # cv2.imshow("something", obs["rgba_img"])
    cv2.waitKey(1)
