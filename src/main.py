import cv2
import numpy as np

from pyflyt_rail_env import Environment

# env = Environment(render_mode="human")
env = Environment(render_mode=None)

obs, _ = env.reset()
term = False
trunc = False
action = np.zeros(*env.action_space.shape)

while not (term or trunc):
    track_position = env.track_state
    action[0] = 2.0
    action[1] = track_position[0] * 3.0
    action[2] = track_position[1] * 3.0
    action[3] = 1.0 - env.drone.state[-1][-1]

    obs, rew, term, trunc, info = env.step(action)
    print(env.step_count)

    cv2.imshow("something", obs["seg_img"][..., 0].astype(np.uint8) * 255)
    # cv2.imshow("something", obs["rgba_img"])
    cv2.waitKey(1)
