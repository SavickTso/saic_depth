import os
import torch

import numpy as np
from PIL import Image

# ROOT = '/Vol1/dbstore/datasets/depth_completion/Matterport3D/'
# ROOT = "/Vol0/user/d.senushkin/datasets/matterport3d"
ROOT = "/root/saic_depth"


class Matterport:
    def __init__(self, root=ROOT, split="train", transforms=None):
        self.transforms = transforms
        self.data_root = os.path.join(root, "data")
        self.split_file = os.path.join(root, "splits", split + ".txt")
        self.data_list = self._get_data_list(self.split_file)
        self.color_name, self.depth_name, self.render_name = [], [], []
        self.normal_name = []

        self._load_data()

    def _load_data(self):
        for x in os.listdir(self.data_root):
            scene = os.path.join(self.data_root, x)
            raw_depth_scene = os.path.join(scene, "undistorted_depth_images")
            render_depth_scene = os.path.join(scene, "render_depth")

            for y in os.listdir(raw_depth_scene):
                (
                    valid,
                    resize_count,
                    one_scene_name,
                    num_1,
                    num_2,
                    png,
                ) = self._split_matterport_path(y)
                if valid == False or png != "png" or resize_count != 1:
                    continue
                data_id = (x, one_scene_name, num_1, num_2)
                if data_id not in self.data_list:
                    continue
                raw_depth_f = os.path.join(raw_depth_scene, y)
                render_depth_f = os.path.join(
                    render_depth_scene, y.split(".")[0] + "_mesh_depth.png"
                )
                color_f = os.path.join(
                    scene,
                    "undistorted_color_images",
                    f"resize_{one_scene_name}_i{num_1}_{num_2}.jpg",
                )
                est_normal_f = os.path.join(
                    scene,
                    "estimate_normal",
                    f"resize_{one_scene_name}_d{num_1}_{num_2}_normal_est.png",
                )

                self.depth_name.append(raw_depth_f)
                self.render_name.append(render_depth_f)
                self.color_name.append(color_f)
                self.normal_name.append(est_normal_f)

    def _get_data_list(self, filename):
        with open(filename, "r") as f:
            content = f.read().splitlines()
        data_list = []
        for ele in content:
            left, _, right = ele.split("/")
            (
                valid,
                resize_count,
                one_scene_name,
                num_1,
                num_2,
                png,
            ) = self._split_matterport_path(right)
            if valid == False:
                print(f"Invalid data_id in datalist: {ele}")
            data_list.append((left, one_scene_name, num_1, num_2))
        return set(data_list)

    def _split_matterport_path(self, path):
        try:
            left, png = path.split(".")
            lefts = left.split("_")
            resize_count = left.count("resize")
            one_scene_name = lefts[resize_count]
            num_1 = lefts[resize_count + 1][-1]
            num_2 = lefts[resize_count + 2]
            return True, resize_count, one_scene_name, num_1, num_2, png
        except Exception as e:
            print(e)
            return False, None, None, None, None, None

    def __len__(self):
        return len(self.depth_name)

    def __getitem__(self, index):
        color = (
            np.array(Image.open(self.color_name[index])).transpose([2, 0, 1]) / 255.0
        )
        render_depth = np.array(Image.open(self.render_name[index])) / 4000.0
        depth = np.array(Image.open(self.depth_name[index])) / 4000.0

        normals = np.array(Image.open(self.normal_name[index])).transpose([2, 0, 1])
        normals = (normals - 90.0) / 180.0

        mask = np.zeros_like(depth)
        mask[np.where(depth > 0)] = 1

        print(
            "the depth filename of the index {} is {} ".format(
                index, self.depth_name[index]
            )
        )

        return {
            "color": torch.tensor(color, dtype=torch.float32),
            "raw_depth": torch.tensor(depth, dtype=torch.float32).unsqueeze(0),
            "mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            "normals": torch.tensor(normals, dtype=torch.float32).unsqueeze(0),
            "gt_depth": torch.tensor(render_depth, dtype=torch.float32).unsqueeze(0),
            "index": torch.tensor(index, dtype=torch.int32),
        }
