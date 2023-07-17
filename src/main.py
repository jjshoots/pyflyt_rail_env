import cv2
import numpy as np

from pyflyt_rail_env import Environment

env = Environment(render_mode="human")
# env = Environment(render_mode=None)

obs, _ = env.reset()
action = np.zeros(*env.action_space.shape)

for i in range(9999999):
    track_position = env.track_state()
    action[0] = 3.0
    action[1] = track_position[0] * 2.0
    action[2] = track_position[1] * 2.0
    action[3] = 1.5 - env.drone.state[-1][-1]

    obs, rew, term, trunc, info = env.step(action)

    # cv2.imshow("something", obs["rgba_img"])
    segimg = obs["seg_img"] * 255
    segimg = segimg.astype(np.uint8)
    cv2.imshow("something", segimg)
    cv2.waitKey(1)

    if term or trunc:
        env.close()
        exit()
