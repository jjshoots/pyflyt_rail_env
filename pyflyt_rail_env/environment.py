"""QuadX Railway Environment with domain randomization."""
from __future__ import annotations

import math
import os
import random

import gymnasium
import numpy as np
from gymnasium import spaces
from PyFlyt.core import obj_collision, obj_visual
from PyFlyt.core.aviary import Aviary

from .MultiRailList import Rail


class Environment(gymnasium.Env):
    """
    QuadX Railway Environment with domain randomization.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_duration_seconds: int = 60,
        agent_hz: int = 10,
        render_mode: None | str = None,
        target_height: float = 1.5,
        target_velocity: float = 1.0,
        max_yaw_rate: float = 1.5,
        corridor_height: float = 5.0,
        corridor_width: float = 7.0,
        corridor_max_angle: float = np.pi / 2.0,
        cam_resolution: tuple[int, int] = (64, 64),
        cam_FOV_degrees: int = 145,
        cam_angle_degrees: int = 0,
    ):
        """__init__.

        Args:
            max_duration_seconds (int): max_duration_seconds
            agent_hz (int): agent_hz
            render_mode (None | str): render_mode
            target_height (float): target_height
            target_velocity (float): target_velocity
            max_yaw_rate (float): max_yaw_rate
            corridor_height (float): corridor_height
            corridor_width (float): corridor_width
            corridor_max_angle (float): corridor_max_angle
            cam_resolution (tuple[int, int]): cam_resolution
            cam_FOV_degrees (int): cam_FOV_degrees
            cam_angle_degrees (int): cam_angle_degrees
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
            low=np.array([-1.0, -max_yaw_rate]),
            high=np.array([1.0, max_yaw_rate]),
            dtype=np.float64,
        )

        # observation space
        attitude_shape = 4 + self.action_space.shape[0]
        self.observation_space = spaces.Dict(
            {
                "attitude": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(attitude_shape,), dtype=np.float64
                ),
                "seg_img": spaces.Box(
                    low=0.0, high=1.0, shape=(*cam_resolution, 2), dtype=np.uint8
                ),
            }
        )

        """ ENVIRONMENT CONSTANTS """
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

        # where the model files are located
        self.rails_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "./models/rails/"
        )
        self.obstacle_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "./models/obstacles/"
        )

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
        self.previous_action = np.zeros(*self.action_space.shape)
        self.action = np.zeros(*self.action_space.shape)
        self.step_count = 0

        # initialize the infos
        self.infos = dict()

        # for reward tracking
        self.distance = 0.0
        self.previous_distance = -math.inf

        # initialize the aviary
        drone_options = dict()
        drone_options["model_dir"] = os.path.join(os.path.dirname(__file__), "models/")
        drone_options["drone_model"] = "modded_drone"
        drone_options["use_camera"] = True
        drone_options["use_gimbal"] = True
        drone_options["camera_resolution"] = self.cam_resolution
        drone_options["camera_FOV_degrees"] = self.cam_FOV_degrees + np.random.randint(-15, 15)
        drone_options["camera_angle_degrees"] = -self.cam_angle_degrees + np.random.randint(-15, 15)
        self.flight_height = self.target_height + (np.random.random() - 0.5)
        start_pos = np.array([[3.0, np.random.randn(), self.flight_height]])
        start_orn = np.array([[0.0, 0.0, np.random.randn()]])
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

        # randomly jitter the height of the rail
        # rail_height = (np.random.rand(1) * 0.25).item()
        start_pos = np.array([0, 0, -0.14])

        # apply a random rotation to the rail
        rail_rotation = ((np.random.rand(1) - 0.5) * np.pi * 0.25).item()
        start_orn = np.array([0.5 * math.pi, 0, -0.5 * np.pi + rail_rotation])

        self.rail = Rail(
            p=self.aviary,
            start_pos=start_pos,
            start_orn=start_orn,
            visual_ids=self.rail_mesh,
        )

        # initialize the track state
        self.track_state = np.array([0.0, 0.0])

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
        - lin_vel (vector of 3 values)
        - height (one scalar)
        - previous_action (vector of 4 values)
        """
        # get the relevant states
        raw_state = self.aviary.state(0)
        lin_vel = raw_state[2]
        height = raw_state[-1][-1]

        # combine everything
        self.state["attitude"] = np.array([*lin_vel, height, *self.action])

        # grab the segmentation image, all boolean dtype
        rail_seg = np.isin(self.drone.segImg, self.rail.rail_ids)
        obstacle_seg = np.isin(self.drone.segImg, self.rail.obstacle_ids)
        total_seg_img = np.concatenate([rail_seg, obstacle_seg], axis=-1)
        self.state["seg_img"] = total_seg_img

        # add 0.1% salt and 5% pepper noise, all boolean dtype
        # salt = np.random.random(size=total_seg_img.shape) > 0.999
        # pepper = np.random.random(size=total_seg_img.shape) > 0.95
        # self.state["seg_img"] = (total_seg_img | salt) & ~pepper

    def compute_track_state(self):
        """
        This returns the position of the track relative to the drone as a [pos, orn] 2 value array.
        """
        # grab the closest rail and compute the angle and vector
        segment = self.rail.closest(self.drone.state[-1])
        seg_dist = segment.base_pos[:2] - self.drone.state[-1][:2]
        seg_angle = np.arctan2(*(segment.end_pos[:2] - segment.base_pos[:2])[::-1])

        # compute the angle and distance relative to the drone
        angle = seg_angle - self.drone.state[1][-1]
        distance = seg_dist[1] * np.cos(seg_angle) - seg_dist[0] * np.sin(seg_angle)

        self.track_state = np.array([distance, angle])

    def compute_term_trunc_reward(self):
        """compute_term_trunc_reward."""

        # drift penalty
        drift_penalty = self.track_state[0] ** 2
        drift_penalty *= 3.0

        # yaw penalty
        yaw_penalty = self.track_state[1] ** 2
        yaw_penalty *= 3.0

        # collision penalty
        collision = np.any(self.aviary.contact_array)
        collision_penalty = 500.0 * collision

        # target loss penalty
        target_loss = np.abs(self.track_state[0]) > self.corridor_width
        target_loss |= np.abs(self.track_state[1]) > self.corridor_max_angle
        target_loss_penalty = 500.0 * target_loss

        # too low
        too_low = self.drone.state[-1][-1] < 0.5
        too_low_penalty = 500.0 * too_low

        # terminate run penalty
        stop_run = self.action[0] < 0.5
        stop_run &= np.linalg.norm(self.drone.state[-2]) < 1.0
        stop_run_penalty = 100.0 * stop_run

        # sum up all rewards
        self.reward += 10.0
        self.reward -= (
            +drift_penalty
            + yaw_penalty
            + collision_penalty
            + target_loss_penalty
            + too_low_penalty
            + stop_run_penalty
        )

        # handle termination truncation
        # terminate on:
        # - target loss
        # - collisions
        # - drifted too far
        # - drone slowed to a close
        self.termination |= target_loss
        self.termination |= collision
        self.termination |= too_low
        self.termination |= stop_run
        self.truncation |= self.step_count > self.max_steps

        self.infos["target_loss"] = target_loss
        self.infos["collision"] = collision
        self.infos["run_stopped"] = stop_run
        self.infos["too_low"] = too_low

    def compute_setpoint(self, action: np.ndarray) -> np.ndarray:
        """Computes the setpoint to give the drone given the agent's action.

        Args:
            action (np.ndarray):
                - [stop/go, yaw_rate]

        Returns:
            tuple[np.ndarray, bool]: the setpoint and whether to terminate the process
        """
        setpoint = np.zeros((4,), dtype=np.float64)

        # override the forward function with a boolean
        setpoint[0] = (action[0] > 0.5) * self.target_velocity

        # don't have sideways drift
        setpoint[1] = 0.0

        # passthrough yaw
        setpoint[2] = action[-1]

        # maintain constant height
        setpoint[3] = (self.flight_height - self.drone.state[-1][-1])

        return setpoint

    def step(self, action: np.ndarray):
        """Steps the environment.

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info
        """
        # unsqueeze the action to be usable in aviary
        self.previous_action = self.action.copy()
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

        # spawn in some obstacles
        self.spawn_obstacle()
        self.aviary.register_all_new_bodies()

        # increment step count
        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.infos

    def spawn_obstacle(self):
        # handle the rail bounds
        spawn_direction = self.rail.handle_rail_bounds(self.drone.state[-1])

        # if nothing spawned just return
        if spawn_direction < 0:
            return

        for _ in range(1):
            continue
            # 20% chance that an obstacle will appear
            if np.random.random() < 0.8:
                continue

            # random chance between box and arch
            chance = np.random.randint(2)
            pos_offset = (np.random.rand(3) - 0.5) * 4.0

            if chance == 0:
                # spawn a box
                collision_id = self.aviary.createCollisionShape(
                    shapeType=self.aviary.GEOM_BOX,
                    halfExtents=np.random.rand(3) * 3.0,
                )
            else:
                # spawn the arch
                mesh_scale = np.random.rand(3) + 0.5
                collision_id = obj_collision(
                    self.aviary,
                    self.obstacle_dir + "arch.obj",
                    concave=True,
                    meshScale=mesh_scale,
                )

            self.rail.tail.add_obstacle(collision_id, pos_offset=pos_offset)

        self.rail.update_obstacle_ids()

    def initialize_common_meshes(self):
        """initialize_common_meshes."""
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
