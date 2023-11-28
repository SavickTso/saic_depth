import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from saic_depth_completion.config import get_default_config
from saic_depth_completion.metrics import SSIM, DepthL1Loss, DepthL2Loss, DepthRel, Miss
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.utils import visualize_nogt
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.snapshoter import Snapshoter

# import keyboard
# import pyzed.sl as sl
# import numpy as np

# def zed_capture_once():
#     zed = sl.Camera()

#     init_params = sl.InitParameters()
#     init_params.camera_resolution = sl.RESOLUTION.HD1080
#     init_params.camera_fps = 60
#     init_params.depth_mode = sl.DEPTH_MODE.ULTRA
#     init_params.coordinate_units = sl.UNIT.MILLIMETER

#     # Open the camera
#     err = zed.open(init_params)
#     if err != sl.ERROR_CODE.SUCCESS:
#         print("Failed to open the camera")
#         return

#     # Create image and depth objects
#     image = sl.Mat()
#     depth = sl.Mat()

#     # Capture 10 color and depth images with a 0.8-second time gap

#     if zed.grab() == sl.ERROR_CODE.SUCCESS:
#         # Check if the "k" key is pressed
#         if keyboard.is_pressed("k"):
#             print("Start capturing...")
#         else:
#             print("Press 'k' to continue...")
#             keyboard.wait("k")

#         time.sleep(0.5)
#         zed.retrieve_image(image, sl.VIEW.LEFT)
#         zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
#         depth_array = depth.get_data()
#         print("depth data shape ", depth_array.shape)
#         depth_array_flat = depth_array.flatten()
#         flat_indices = np.argsort(depth_array_flat)
#         print("max depth is ", depth_array_flat[flat_indices[-2]])

#     zed.close()
#     return image, depth


def inference(model, batch, metrics, save_dir="", logger=None):
    model.eval()
    metrics_meter = AggregatedMeter(metrics, maxlen=20)

    metrics_meter.reset()

    batch = model.preprocess(batch)

    pred = model(batch)

    with torch.no_grad():
        post_pred = model.postprocess(pred)
        if save_dir:
            B = batch["color"].shape[0]
            for it in range(B):
                fig = visualize_nogt.figure(
                    batch["color"][it],
                    batch["raw_depth"][it],
                    batch["mask"][it],
                    post_pred[it],
                    close=True,
                )
                fig.savefig(
                    os.path.join(save_dir, "result_single.png"),
                    dpi=fig.dpi,
                )


def main():
    parser = argparse.ArgumentParser(description="Some training params.")

    parser.add_argument(
        "--default_cfg",
        dest="default_cfg",
        type=str,
        default="arch0",
        help="Default config",
    )
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--save_dir", default="", type=str, help="Save dir for predictions"
    )
    parser.add_argument(
        "--weights", default="", type=str, metavar="FILE", help="path to config file"
    )

    args = parser.parse_args()

    cfg = get_default_config(args.default_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MetaModel(cfg, device)

    logger = setup_logger()

    snapshoter = Snapshoter(model, logger=logger)
    snapshoter.load(args.weights)

    metrics = {
        "mse": DepthL2Loss(),
        "mae": DepthL1Loss(),
        "d105": Miss(1.05),
        "d110": Miss(1.10),
        "d125_1": Miss(1.25),
        "d125_2": Miss(1.25**2),
        "d125_3": Miss(1.25**3),
        "rel": DepthRel(),
        "ssim": SSIM(),
    }

    color = plt.imread("/root/saic_depth/data/0000000000.png")
    depth = plt.imread("/root/saic_depth/data/0000000000_depth.png")
    mask = np.zeros_like(depth)
    mask[np.where(depth > 0)] = 1
    # normals = plt.imread("/root/saic_depth/data/0000000000_normal.png")
    index = 0

    batch = {
        "color": torch.tensor(color, dtype=torch.float32),
        "raw_depth": torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        "mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        # "normals": torch.tensor(normals, dtype=torch.float32).unsqueeze(0),
        # "gt_depth": torch.tensor(render_depth, dtype=torch.float32).unsqueeze(0),
        "index": torch.tensor(index, dtype=torch.int32),
    }

    inference(
        model,
        batch,
        save_dir=args.save_dir,
        logger=logger,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
