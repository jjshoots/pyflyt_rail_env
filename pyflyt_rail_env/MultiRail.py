import math

import numpy as np
from pybullet_utils import bullet_client
from PyFlyt.core.load_objs import loadOBJ


class MultiRail:
    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        rail_mesh_ids: np.ndarray,
    ):
        self.p = p
        self.rail_mesh_ids = rail_mesh_ids
        self.tex_id = None

        rail = SingleRail(p, start_pos, start_orn, self.rail_mesh_ids, 0)

        self.Ids = [rail.Id]
        self.head = rail.get_end(0)
        self.tail = rail.get_end(1)

    def handle_rail_bounds(self, drone_xy: np.ndarray, direction: int = -1) -> int:
        """
        extends and deletes rails on the fly
            drone_xy: 2d array for drone x and y positions
            direction:
                -2 for no spawn
                -1 for spawn random if required
                +0 for spawn straight rail
                +1 for spawn left
                +2 for spawn right

        returns the spawn action that's been taken, can be -2, 0, 1, 2 but not -1
        """
        head_delete = []
        tail_delete = []
        dis2head = np.sum((self.head.base_pos[:2] - drone_xy) ** 2) ** 0.5
        dis2tail = np.sum((self.tail.base_pos[:2] - drone_xy) ** 2) ** 0.5

        # delete the head if it's too far and get the new one
        if dis2head > 20:
            head_delete, self.head = self.head.delete(0)

        # if the tail is too far away, just delete it
        if dis2tail > 100:
            tail_delete, self.tail = self.tail.delete(1)

        self.Ids = [
            i for i in self.Ids if (i not in head_delete and i not in tail_delete)
        ]

        # if there is no need to spawn more, just return
        if direction == -2:
            return direction

        # create new tail if it's too near if allowed
        if dis2tail < 40:
            # if don't have spawn direction, just random
            direction = np.random.randint(0, 3) if direction == -1 else direction

            # spawn a new tail
            self.tail.add_child(self.rail_mesh_ids, direction)
            self.tail = self.tail.get_end(1)
            self.Ids.append(self.tail.Id)

            # change texture if we have a texture
            if self.tex_id is not None:
                self.p.changeVisualShape(self.tail.Id, -1, textureUniqueId=self.tex_id)

            return direction
        else:
            return -2

    def change_texture(self, tex_id):
        self.tex_id = tex_id
        for id in self.Ids:
            self.p.changeVisualShape(id, -1, textureUniqueId=self.tex_id)


class SingleRail:
    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        rail_mesh_ids: np.ndarray,
        spawn_id: int,
        parent=None,
    ):
        self.p = p

        # values are obtained from the readme in the rails models folder
        ROT = 0.1 * math.pi
        c, s = np.cos(-start_orn[-1]), np.sin(-start_orn[-1])
        rot_mat = np.array([[c, -s], [s, c]]).T

        if spawn_id == 0:
            end_offset = np.matmul(rot_mat, np.array([0, 10.125]))
            end_offset = np.array([*end_offset, 0.0])

            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = self.start_pos + 2 * end_offset
            self.end_orn = self.start_orn
            self.base_pos = self.start_pos + end_offset
            self.base_orn = self.start_orn
        elif spawn_id == 1:
            end_offset = np.matmul(rot_mat, np.array([-2.209, 20.186]))
            end_offset = np.array([*end_offset, 0.0])
            base_offset = np.matmul(rot_mat, np.array([0.0, 10.061]))
            base_offset = np.array([*base_offset, 0.0])

            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = self.start_pos + end_offset
            self.end_orn = self.start_orn + np.array([0, 0, ROT])
            self.base_pos = self.start_pos + base_offset
            self.base_orn = self.start_orn
        elif spawn_id == 2:
            end_offset = np.matmul(rot_mat, np.array([2.209, 20.186]))
            end_offset = np.array([*end_offset, 0.0])
            base_offset = np.matmul(rot_mat, np.array([0.0, 10.061]))
            base_offset = np.array([*base_offset, 0.0])

            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = self.start_pos + end_offset
            self.end_orn = self.start_orn + np.array([0, 0, -ROT])
            self.base_pos = self.start_pos + base_offset
            self.base_orn = self.start_orn
        else:
            print("IMPORTING UNKNOWN RAIL OBJECT")
            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = start_pos
            self.end_orn = start_orn
            self.base_pos = start_pos
            self.base_orn = start_orn

        self.Id = loadOBJ(
            self.p,
            visualId=rail_mesh_ids[spawn_id],
            basePosition=self.base_pos,
            baseOrientation=self.p.getQuaternionFromEuler(self.base_orn),
        )

        # linked = [parent, child]
        self.linked: list[SingleRail | None] = [None, None]
        self.linked[0] = parent

    def add_child(self, rail_mesh_ids, spawn_id):
        """adds a single child to the end of the rail"""
        self.linked[1] = SingleRail(
            self.p, self.end_pos, self.end_orn, rail_mesh_ids, spawn_id, self
        )

    def delete(self, direction: int):
        """
        deletes self and all connected links in direction, 0 for parent, 1 for child.
        <-- 0, 1 -->
        start - node - parent(opp) - self - child(opp) - node - end
        """
        deleted = []
        node = self

        # traverse to the end of the chain
        while node.linked[direction] is not None:
            node = node.linked[direction]

        while True:
            # if the opposite is None, we have no more rails on this line
            if node.linked[1 - direction] is None:
                break

            # if we have deleted to opposite, break out of loop
            if node.Id == self.linked[1 - direction].Id:
                break

            # delete the node model and record the deletion
            # then move up the chain by a step and dereference
            self.p.removeBody(node.Id)
            deleted.append(node.Id)
            node = node.linked[1 - direction]
            node.linked[direction] = None

        # return list of deletions, and the opposite node
        return deleted, node

    def get_end(self, direction: int):
        """gets the end of the track, depending on dir, 0 for parent, 1 for child"""
        node = self
        while node.linked[direction] is not None:
            node = node.linked[direction]

        return node
