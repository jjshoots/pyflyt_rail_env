"""QuadX Railway Environment with domain randomization."""
from __future__ import annotations

import glob
import math
import os

import gymnasium
import numpy as np
import numpy.polynomial.polynomial as polynomial
from gymnasium import spaces
from PyFlyt.core.aviary import Aviary
from PyFlyt.core.load_objs import obj_collision, obj_visual

from .MultiRailList import Rail


class Environment(gymnasium.Env):
    """
    QuadX Railway Environment with domain randomization.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_duration_seconds: int = 60,
        agent_hz: int = 30,
        render_mode: None | str = None,
        spawn_height: float = 1.5,
        target_height: float = 1.0,
        target_velocity: float = 3.0,
        max_velocity: float = 3.0,
        max_yaw_rate: float = np.pi,
        corridor_height: float = 10.0,
        corridor_width: float = 5.0,
        corridor_max_angle: float = np.pi / 5.0,
        cam_resolution: tuple[int, int] = (64, 64),
        cam_FOV_degrees: int = 145,
        cam_angle_degrees: int = 70,
        update_textures_seconds: int = 10,
    ):
        """__init__.

        Args:
            max_duration_seconds (int): max_duration_seconds
            agent_hz (int): agent_hz
            render_mode (None | str): render_mode
            spawn_height (float): spawn_height
            target_height (float): target_height
            target_velocity (float): target_velocity
            max_velocity (float): max_velocity
            max_yaw_rate (float): max_yaw_rate
            corridor_height (float): corridor_height
            corridor_width (float): corridor_width
            corridor_max_angle (float): corridor_max_angle
            cam_resolution (tuple[int, int]): cam_resolution
            cam_FOV_degrees (int): cam_FOV_degrees
            cam_angle_degrees (int): cam_angle_degrees
            update_textures_seconds (int): update_textures_seconds
        """
        if 120 % agent_hz != 0:
            lowest = int(120 / (int(120 / agent_hz) + 1))
            highest = int(120 / int(120 / agent_hz))
            raise AssertionError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        if render_mode is not None:
            assert (
                render_mode in self.metadata["render_modes"]
            ), f"Invalid render mode {render_mode}, only {self.metadata['render_modes']} allowed."
        self.render_mode = render_mode

        """GYMNASIUM STUFF"""
        # action space
        self.action_space = spaces.Box(
            low=np.array([-max_velocity, -max_velocity, -max_yaw_rate, -max_velocity]),
            high=np.array([max_velocity, max_velocity, max_yaw_rate, max_velocity]),
            dtype=np.float64,
        )

        # observation space
        attitude_shape = 6 + self.action_space.shape[0]
        self.observation_space = spaces.Dict(
            {
                "attitude": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(attitude_shape,), dtype=np.float64
                ),
                "rgba_img": spaces.Box(
                    low=0.0, high=255.0, shape=(4, *cam_resolution), dtype=np.uint8
                ),
                "seg_img": spaces.Box(
                    low=0.0, high=1024.0, shape=(4, *cam_resolution), dtype=np.uint8
                ),
            }
        )

        """ ENVIRONMENT CONSTANTS """
        self.spawn_height = spawn_height
        self.target_height = target_height
        self.target_velocity = target_velocity
        self.corridor_height = corridor_height
        self.corridor_width = corridor_width
        self.corridor_max_angle = corridor_max_angle
        self.cam_resolution = cam_resolution
        self.cam_FOV_degrees = cam_FOV_degrees
        self.cam_angle_degrees = cam_angle_degrees
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        self.update_textures_step = update_textures_seconds * agent_hz

        # where the model files are located
        self.rails_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "./models/rails/"
        )
        self.clutter_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "./models/clutter/"
        )
        tex_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "./models/textures/images/"
        )
        self.texture_paths = glob.glob(
            os.path.join(tex_dir, "**", "*.jpg"), recursive=True
        )

        # form the array for inverse projection later
        seg_centre = (self.cam_resolution[1] + 1) / 2
        rad_per_pixel = (cam_FOV_degrees / 180 * math.pi) / self.cam_resolution[1]
        xspace = np.arange(self.cam_resolution[0], 0, -1)
        yspace = np.arange(self.cam_resolution[1], 0, -1)
        angle_array = np.stack(np.meshgrid(xspace, yspace), axis=-1) - seg_centre
        angle_array *= rad_per_pixel
        angle_array[:, :, 1] += cam_angle_degrees / 180.0 * math.pi
        angle_array = angle_array.reshape(-1, 2)
        y = np.sin(angle_array[:, 1]) / abs(np.cos(angle_array[:, 1]))
        x = np.tan(angle_array[:, 0]) / abs(np.cos(angle_array[:, 1]))
        self.inv_proj = np.stack((x, y), axis=-1)
        # plt.scatter(self.inv_proj[:, 0], self.inv_proj[:, 1])
        # plt.show()
        # exit()

        """ INITIALIZE """
        self.aviary: Aviary
        self.reset()

    def reset(self, seed=None, options=dict()):
        """reset.

        Args:
            seed:
            options:
        """
        super().reset(seed=seed)

        # if we already have an env, disconnect from it
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

        # reset parameters
        self.termination = False
        self.truncation = False
        self.state = dict()
        self.action = np.zeros(*self.action_space.shape)
        self.step_count = 0

        # initialize the infos
        self.infos = dict()

        # for reward tracking
        self.distance = 0.0
        self.previous_distance = -math.inf

        # initialize the aviary
        drone_options = dict()
        drone_options["drone_model"] = "primitive_drone"
        drone_options["use_camera"] = True
        drone_options["use_gimbal"] = True
        drone_options["camera_resolution"] = self.cam_resolution
        drone_options["camera_FOV_degrees"] = self.cam_FOV_degrees
        drone_options["camera_angle_degrees"] = -self.cam_angle_degrees
        start_pos = np.array([[1.0, 0.0, self.spawn_height]])
        start_orn = np.array([[0.0, 0.0, 0.0]])
        self.aviary = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type="quadx",
            drone_options=drone_options,
            render=self.render_mode is not None,
            world_scale=10.0,
        )
        self.aviary.set_mode(5)

        # grab the pointer to the first drone
        self.drone = self.aviary.drones[0]

        # preload the common meshes
        self.initialize_common_meshes()

        # start rails, the first rail in the list is the main rail to follow
        self.rails: list[Rail] = []
        start_pos = np.array([0, 0, 0])
        start_orn = np.array([0.5 * math.pi, 0, -0.5 * math.pi])
        self.rails.append(
            Rail(
                p=self.aviary,
                start_pos=start_pos,
                start_orn=start_orn,
                visual_ids=self.rail_mesh,
            )
        )

        # initialize the track state
        self.track_state = np.array([0.0, 0.0])

        # update all textures
        self.update_textures()

        # wait for env to stabilize
        for _ in range(10):
            self.aviary.step()

        # get the state
        self.compute_state()

        # get the debug camera parameters
        if self.render_mode is not None:
            self.camera_parameters = self.aviary.getDebugVisualizerCamera()

        return self.state, dict()

    def close(self):
        """Disconnects the internal Aviary."""
        # if we already have an env, disconnect from it
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

    def compute_state(self):
        """Computes the state of the current timestep.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - lin_vel (vector of 3 values)
        - previous_action (vector of 4 values)
        """
        # get the relevant states
        raw_state = self.aviary.state(0)
        ang_vel = raw_state[0]
        lin_vel = raw_state[2]

        # combine everything
        self.state["attitude"] = np.array([*ang_vel, *lin_vel, *self.action])

        # grab the image
        self.state["seg_img"] = np.isin(self.drone.segImg, self.rails[0].rail_ids)
        self.state["rgba_img"] = self.drone.rgbaImg.astype(np.uint8)

    def compute_track_state(self):
        """
        This returns the position of the track relative to the drone as a [pos, orn] 2 value array.
        """
        if np.sum(self.state["seg_img"]) > self.cam_resolution[1]:
            proj = (
                self.inv_proj[self.state["seg_img"].flatten()]
                * self.drone.state[-1][-1]
            )

            poly = polynomial.Polynomial.fit(proj[:, 1], proj[:, 0], 2).convert(
                domain=(-1, 1)
            )
            pos = polynomial.polyval(1.0, [*poly])
            orn = math.atan(polynomial.polyval(1.0, [*poly.deriv()]))

            # plt.scatter(proj[:, 1], proj[:, 0])
            # plt.plot(*poly.linspace(n=100, domain=(0, np.max(proj[:, 1]))), "y")
            # plt.show()
            # exit()

            # normalize
            state = np.array([pos, orn])
            self.track_state = np.clip(state, -0.99999, 0.99999)

        else:
            self.track_state = np.array([0.0, 0.0])

    def compute_term_trunc_reward(self):
        """compute_term_trunc_reward.
        """

        # drift penalty
        drift_penalty = (self.track_state[0]) ** 2
        drift_penalty *= 5.0

        # yaw penalty
        yaw_penalty = (self.track_state[1]) ** 2
        yaw_penalty *= 5.0

        # height penalty
        height_penalty = (self.drone.state[-1][-1] - self.target_height) ** 2
        height_penalty *= 5.0

        # collision reward is negative of collision
        collision_penalty = np.any(self.aviary.contact_array)
        collision_penalty *= 1000.0

        # target loss penalty
        target_loss = self.state["seg_img"].sum() < self.cam_resolution[0]
        target_loss_penalty = 1000.0 * target_loss

        # sum up all rewards
        self.reward += 5.0
        self.reward -= (
            + drift_penalty
            + yaw_penalty
            + height_penalty
            + collision_penalty
            + target_loss_penalty
        )

        # handle termination truncation
        # terminate on:
        # - target loss
        # - collisions
        # - drifted too far
        # - drone slowed to a close
        self.termination |= target_loss
        self.termination |= np.any(self.aviary.contact_array)
        self.termination |= np.abs(self.track_state[0]) > self.corridor_width
        self.termination |= self.drone.state[-1][0] < 0.1
        self.truncation |= self.step_count > self.max_steps

    def compute_setpoint(self, action: np.ndarray) -> np.ndarray:
        """Computes the setpoint to give the drone given the agent's action.

        Args:
            action (np.ndarray): action

        Returns:
            tuple[np.ndarray, bool]: the setpoint and whether to terminate the process
        """
        setpoint = action.copy()

        # override the forward function with a boolean
        setpoint[0] = (action[0] > 0.5) * self.target_velocity

        # don't go higher if we're at the ceiling
        if self.drone.state[-1][-1] >= self.corridor_height:
            setpoint[2] = np.clip(setpoint[2], a_min=None, a_max=0.0)

        return setpoint

    def step(self, action: np.ndarray):
        """Steps the environment.

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info
        """
        # unsqueeze the action to be usable in aviary
        self.action = action.copy()
        self.aviary.set_setpoint(0, self.compute_setpoint(action))

        # step through env, the internal env updates a few steps before the outer env
        self.reward = 0.0
        for _ in range(self.env_step_ratio):
            # if already ended, just complete the loop
            if self.termination or self.truncation:
                break
            self.aviary.step()
            self.compute_state()
            self.compute_track_state()
            self.compute_term_trunc_reward()

        # handle the rails and clutter
        spawn_direction = self.rails[0].handle_rail_bounds(self.drone.state[-1])
        if spawn_direction == 0 and False:
            self.rails[0].tail.add_clutter(
                self.tunnel_visual,
                self.tunnel_collision,
                np.array([0, 10.125, 0]),
                np.array([0, 0, 0]),
            )

        # change the texture of the floor
        if self.step_count % self.update_textures_step == 1:
            self.update_textures()

        # increment step count
        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.infos

    def initialize_common_meshes(self):
        """initialize_common_meshes.
        """
        # rail meshes
        self.rail_mesh = np.ones(3) * -1
        self.rail_mesh[0] = obj_visual(
            self.aviary, self.rails_dir + "rail_straight.obj"
        )
        self.rail_mesh[1] = obj_visual(
            self.aviary, self.rails_dir + "rail_turn_left.obj"
        )
        self.rail_mesh[2] = obj_visual(
            self.aviary, self.rails_dir + "rail_turn_right.obj"
        )

        # clutter meshes
        self.tunnel_visual = obj_visual(self.aviary, self.clutter_dir + "tunnel.obj")

        # collision meshes for the clutter
        self.tunnel_collision = obj_collision(
            self.aviary, self.clutter_dir + "tunnel.obj", concave=True
        )

    def update_textures(self):
        """
        randomly change the texture of the env
        25% chance of the rail being same texture as floor
        25% chance of clutter being same texture as rails
        25% chance of rail, floor, and clutter being same texture
        25% chance of all different
        """
        return
        chance = np.random.randint(4)

        if chance == 0:
            # rail and floor same, clutter diff
            tex_id = self.get_random_texture()
            for rail in self.rails:
                rail.change_rail_texture(tex_id)
            self.aviary.changeVisualShape(
                self.aviary.planeId, -1, textureUniqueId=tex_id
            )

            tex_id = self.get_random_texture()
            for rail in self.rails:
                rail.change_clutter_texture(tex_id)
        elif chance == 1:
            # clutter and floor same, rail diff
            tex_id = self.get_random_texture()
            for rail in self.rails:
                rail.change_clutter_texture(tex_id)
            self.aviary.changeVisualShape(
                self.aviary.planeId, -1, textureUniqueId=tex_id
            )

            tex_id = self.get_random_texture()
            for rail in self.rails:
                rail.change_rail_texture(tex_id)
        elif chance == 2:
            # all same
            tex_id = self.get_random_texture()
            for rail in self.rails:
                rail.change_rail_texture(tex_id)
                rail.change_clutter_texture(tex_id)
            self.aviary.changeVisualShape(
                self.aviary.planeId, -1, textureUniqueId=tex_id
            )
        else:
            # all same
            tex_id = self.get_random_texture()
            for rail in self.rails:
                rail.change_rail_texture(tex_id)

            tex_id = self.get_random_texture()
            for rail in self.rails:
                rail.change_clutter_texture(tex_id)

            tex_id = self.get_random_texture()
            self.aviary.changeVisualShape(
                self.aviary.planeId, -1, textureUniqueId=tex_id
            )

    def get_random_texture(self) -> int:
        """get_random_texture.

        Args:

        Returns:
            int:
        """
        texture_path = self.texture_paths[
            np.random.randint(0, len(self.texture_paths) - 1)
        ]
        tex_id = -1
        while tex_id < 0:
            tex_id = self.aviary.loadTexture(texture_path)
        return tex_id

    def render(self) -> np.ndarray:
        """render.

        Args:

        Returns:
            np.ndarray:
        """
        _, _, rgbaImg, _, _ = self.aviary.getCameraImage(
            width=640,
            height=480,
            viewMatrix=self.camera_parameters[2],
            projectionMatrix=self.camera_parameters[3],
        )

        rgbaImg = np.asarray(rgbaImg).reshape(480, 640, -1)

        return rgbaImg
