import numpy as np
import pybullet as p
from typing import Optional

# crowd_sim/base/perception_mixin.py
class VLMPerceptionMixin:
    def __init__(self):
        # 上一帧可见的 human id 集合
        self._prev_visible_ids: set[int] = set()
        # 缓存：id -> activity
        self.human_activity_cache: dict[int, str] = {}
        self.scene_type: str | None = None

    def get_num_human_in_fov(self):
        humans_in_view, ids = [], []
        half_fov = self.robot_fov / 2
        for h in self.humans:
            dx, dy = h.px - self.robot.px, h.py - self.robot.py
            dist = (dx**2 + dy**2) ** 0.5
            if dist > self.robot.sensor_range:
                continue
            ang = np.arctan2(dy, dx) - self.robot.theta
            ang = np.arctan2(np.sin(ang), np.cos(ang))   # wrap‑to‑pi
            if abs(ang) <= half_fov:
                humans_in_view.append(h)
                ids.append(h.id)
        return humans_in_view, set(ids)

    def upload_images_to_vlm(self, rgb_image: Optional[bytes] = None):
        wrong_detected_ids = False
        VLM_is_used = False

        # get current visible human ids
        curr_ids, _, _ = self.get_num_human_in_fov()
        curr_ids = set(curr_ids)
        # find new ids in robot FOV
        new_ids = curr_ids - self._prev_visible_ids

        # used to check if any previous human's activity is not detected
        for human in self.humans:
            closest_points = p.getClosestPoints(self.robot.uid, human.uid, distance=100.0, linkIndexA=-1, linkIndexB=-1)
            if closest_points:
                # The closest points are typically in closest_points[0], check its distance
                dist = closest_points[0][8]
            if human.uid in self._prev_visible_ids and human.uid not in curr_ids:
                if human.detected_activity is None and dist < 2.5:
                    #self.human_activity_cache[human.uid] = self.humans[i].detected_activity
                    wrong_detected_ids = True

        if new_ids or wrong_detected_ids:
            hid_list, acts, scenes = self.query_vlm(rgb_image)
            for hid, act in zip(map(int, hid_list), acts):
                self.human_activity_cache[hid] = act
                for h in self.humans:
                    if h.uid == hid and h.uid in curr_ids:
                        h.detected_activity = act
                        self.set_activity_priorities(h)
            if scenes:
                self.scene_type = scenes[0]   # only one scene type is expected
            print(f"VLM Perception: {len(new_ids)} new ids detected, scene type: {self.scene_type}")
            VLM_is_used = True
        # update previous visible ids
        self._prev_visible_ids = curr_ids
        return VLM_is_used
