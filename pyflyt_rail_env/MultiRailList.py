from __future__ import annotations

import math

import numpy as np
from pybullet_utils import bullet_client
from PyFlyt.core.load_objs import loadOBJ


class Rail:
    """Rail."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        visual_ids: np.ndarray = np.array([-1, -1, -1]),
        collision_ids: np.ndarray = np.array([-1, -1, -1]),
    ):
        """__init__.

        Args:
            p (bullet_client.BulletClient): p
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            visual_ids (np.ndarray): visual_ids
            collision_ids (np.ndarray): collision_ids
        """
        self.p = p
        self.visual_ids = visual_ids
        self.collision_ids = collision_ids

        # spawn the first rail segment
        self.rails: list[SingleRail] = []
        self.rails.append(
            SingleRail(
                p=p,
                start_pos=start_pos,
                start_orn=start_orn,
                visual_ids=visual_ids,
                collision_ids=collision_ids,
                direction=0,
            )
        )

        # list of rail and obstacle ids
        self.rail_ids = np.array([self.rails[0].Id], dtype=np.int32)
        self.obstacle_ids = np.array([])

        # array of [n, 3] where 3 is xyz base and end position of each rail segment
        self.rail_base = np.array([self.rails[0].base_pos], dtype=np.float64)
        self.rail_end = np.array([self.rails[0].base_pos], dtype=np.float64)

    @property
    def head(self) -> SingleRail:
        """head.

        Args:

        Returns:
            SingleRail:
        """
        return self.rails[0]

    @property
    def tail(self) -> SingleRail:
        """tail.

        Args:

        Returns:
            SingleRail:
        """
        return self.rails[-1]

    def closest(self, drone_pos: np.ndarray) -> SingleRail:
        """Returns the rail that is closest to the drone

        Args:
            drone_pos (np.ndarray): drone_pos

        Returns:
            SingleRail:
        """
        base_diff = np.linalg.norm(self.rail_base[:, :2] - drone_pos[:2], axis=-1)
        end_diff = np.linalg.norm(self.rail_end[:, :2] - drone_pos[:2], axis=-1)
        closest = np.argmin(base_diff + end_diff)
        return self.rails[closest]

    def change_rail_color(self, rgba: np.ndarray):
        """change_rail_texture.

        Args:
            texture_id (int): texture_id
        """
        for i in self.rail_ids:
            self.p.changeVisualShape(i, -1, rgbaColor=rgba)

    def change_rail_texture(self, texture_id: int):
        """change_rail_texture.

        Args:
            texture_id (int): texture_id
        """
        for i in self.rail_ids:
            self.p.changeVisualShape(i, -1, textureUniqueId=texture_id)

    def change_obstacle_texture(self, texture_id: int):
        """change_obstacle_texture.

        Args:
            texture_id (int): texture_id
        """
        for rails in self.rails:
            for i in rails.obstacle_ids:
                self.p.changeVisualShape(i, -1, textureUniqueId=texture_id)

    def update_obstacle_ids(self):
        """update_obstacle_ids."""
        self.obstacle_ids = np.concatenate([r.obstacle_ids for r in self.rails])

    def handle_rail_bounds(self, drone_pos: np.ndarray, direction: int = -1) -> int:
        """Extends and deletes rails on the fly

        Args:
            drone_pos (np.ndarray): drone_pos
            direction (int): direction
                -2 for no spawn
                -1 for spawn random if required
                +0 for spawn straight rail
                +1 for spawn left
                +2 for spawn right

        Returns:
            int: direction
        """
        head = self.rails[0]
        tail = self.rails[-1]
        head_distance = np.sum((head.base_pos - drone_pos)[:2] ** 2) ** 0.5
        tail_distance = np.sum((tail.base_pos - drone_pos)[:2] ** 2) ** 0.5

        # if the head is too far, delete
        if head_distance > 20:
            self.p.removeBody(self.rail_ids[0])
            for i in self.rails[0].obstacle_ids:
                self.p.removeBody(i)
            self.rail_ids = self.rail_ids[1:]
            self.rail_base = self.rail_base[1:]
            self.rail_end = self.rail_end[1:]
            self.rails.pop(0)

        # if the tail is too near, add a new one
        if tail_distance < 40:
            # if don't have spawn direction, just random
            direction = np.random.randint(0, 3) if direction == -1 else direction

            self.rails.append(
                SingleRail(
                    p=self.p,
                    start_pos=tail.end_pos,
                    start_orn=tail.end_orn,
                    visual_ids=self.visual_ids,
                    collision_ids=self.collision_ids,
                    direction=direction,
                )
            )
            self.rail_ids = np.append(self.rail_ids, self.rails[-1].Id)
            self.rail_base = np.append(
                self.rail_base, [self.rails[-1].base_pos], axis=0
            )
            self.rail_end = np.append(self.rail_end, [self.rails[-1].end_pos], axis=0)

        return direction


class SingleRail:
    """SingleRail."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        visual_ids: np.ndarray,
        collision_ids: np.ndarray,
        direction: int,
    ):
        """__init__.

        Note:
            end_pos defines the end point of the track
            base_pos is somewhere in the middle of it, NOT THE START

        Args:
            p (bullet_client.BulletClient): p
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            visual_ids (np.ndarray): visual_ids
            collision_ids (np.ndarray): collision_ids
            spawn_id (int): spawn_id
        """
        self.p = p

        # values are obtained from the readme in the rails models folder
        ROT = 0.1 * math.pi
        c, s = np.cos(-start_orn[-1]), np.sin(-start_orn[-1])
        rot_mat = np.array([[c, -s], [s, c]]).T

        if direction == 0:
            end_offset = np.matmul(rot_mat, np.array([0, 10.125]))
            end_offset = np.array([*end_offset, 0.0])

            self.end_pos = start_pos + 2 * end_offset
            self.end_orn = start_orn
            self.base_pos = start_pos + end_offset
            self.base_orn = start_orn
        elif direction == 1:
            base_offset = np.matmul(rot_mat, np.array([0.0, 10.061]))
            base_offset = np.array([*base_offset, 0.0])
            end_offset = np.matmul(rot_mat, np.array([-2.209, 20.186]))
            end_offset = np.array([*end_offset, 0.0])

            self.end_pos = start_pos + end_offset
            self.end_orn = start_orn + np.array([0, 0, ROT])
            self.base_pos = start_pos + base_offset
            self.base_orn = start_orn
        elif direction == 2:
            base_offset = np.matmul(rot_mat, np.array([0.0, 10.061]))
            base_offset = np.array([*base_offset, 0.0])
            end_offset = np.matmul(rot_mat, np.array([2.209, 20.186]))
            end_offset = np.array([*end_offset, 0.0])

            self.end_pos = start_pos + end_offset
            self.end_orn = start_orn + np.array([0, 0, -ROT])
            self.base_pos = start_pos + base_offset
            self.base_orn = start_orn
        else:
            print("IMPORTING UNKNOWN RAIL OBJECT")
            self.end_pos = start_pos
            self.end_orn = start_orn
            self.base_pos = start_pos
            self.base_orn = start_orn

        self.Id = loadOBJ(
            self.p,
            visualId=visual_ids[direction],
            collisionId=collision_ids[direction],
            basePosition=self.base_pos,
            baseOrientation=self.p.getQuaternionFromEuler(self.base_orn),
        )

        self.obstacle_ids = []

    def add_obstacle(
        self,
        collision_id: int,
        pos_offset: np.ndarray = np.zeros(3),
        orn_offset: np.ndarray = np.zeros(3),
    ):
        """add_obstacle."""
        # for orientation, only grab the yaw
        base_orn = np.zeros(3)
        base_orn[-1] = self.base_orn[-1]
        orientation = self.p.getQuaternionFromEuler(base_orn + orn_offset)
        self.obstacle_ids.append(
            self.p.createMultiBody(
                baseCollisionShapeIndex=collision_id,
                basePosition=self.base_pos + pos_offset,
                baseOrientation=orientation,
            )
        )
