import matplotlib.pyplot as plt
import numpy as np


def figure(color, raw_depth, mask, pred, close=False):
    fig, axes = plt.subplots(2, 2, figsize=(7, 10))

    color = color.cpu().permute(1, 2, 0)
    raw_depth = raw_depth.cpu()
    mask = mask.cpu()
    pred = pred.detach().cpu()

    vmin = pred.min()
    vmax = pred.max()

    axes[0, 0].set_title("RGB")
    axes[0, 0].imshow((color - color.min()) / (color.max() - color.min()))

    axes[0, 1].set_title("raw_depth")
    img = axes[0, 1].imshow(raw_depth[0], cmap="RdBu_r")
    fig.colorbar(img, ax=axes[0, 1])

    axes[1, 0].set_title("mask")
    axes[1, 0].imshow(mask[0])

    axes[1, 1].set_title("pred")
    img = axes[1, 1].imshow(pred[0], cmap="RdBu_r", vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=axes[1, 1])
    if close:
        plt.close(fig)
    return fig
