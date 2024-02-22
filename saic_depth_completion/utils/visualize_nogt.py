from copy import deepcopy

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def find_average_non_nan(img, x, y, window_size=10):
    average_depth = 0

    if x >= 240 or y >= 300 or x <= 20 or y <= 20:
        return 0

    roi = img[x - window_size : x + window_size, y - window_size : y + window_size]
    average_depth = np.mean(roi) * roi.size / np.count_nonzero(roi)

    return average_depth


def find_nearest_non_nan(img, x, y, ctr):
    """
    Spirally find the nearest non-NAN depth value to represent the matched invalid coordinates
    """
    # Dmap = Dmaptranp.transpose()
    # Spiral search loop
    m = 1
    newdepth = 0
    while m < 10 and img[x][y] == 0.0:
        if x >= 240 or y >= 300 or x <= 20 or y <= 20:
            ctr += 1
            return 0, ctr
        n = 0
        while n <= m:
            y = y + 1 if m % 2 == 1 else y - 1
            n = n + 1
            newdepth = img[x][y]
            if newdepth != 0:
                break
        n = 0
        while n <= m:
            x = x + 1 if m % 2 == 1 else x - 1
            n = n + 1
            newdepth = img[x][y]
            if newdepth != 0:
                break
        m += 1
    if newdepth == 0:
        ctr += 1

    return newdepth, ctr


def figure(color, raw_depth, mask, pred, close=False, saveidx=0):
    fig, axes = plt.subplots(3, 2, figsize=(7, 10))

    color = color.cpu().permute(1, 2, 0)
    raw_depth = raw_depth.cpu()
    mask = mask.cpu()
    pred = pred.detach().cpu()


    img_origin = deepcopy(raw_depth.squeeze(dim=0).numpy())
    img_spiral = deepcopy(img_origin)
    img_average = deepcopy(img_origin)
    ctr = 0
    for i in range(img_origin.shape[0]):
        for j in range(img_origin.shape[1]):
            # i is y and j is x
            if img_origin[i][j] == 0:
                nearest_depth, ctr = find_nearest_non_nan(img_origin, i, j, ctr)
                img_spiral[i][j] = nearest_depth
                avg_depth = find_average_non_nan(img_origin, i, j)
                img_average[i][j] = avg_depth

    vmin = pred.min()
    vmax = pred.max()

    nmax = img_spiral.max()
    nmin = img_spiral.min()


    print("polars", vmax, vmin, nmax, nmin)
    # Normalize the predict depth image
    pred = (pred - vmin) * ((nmax - nmin) / (vmax - vmin)) + nmin
    random_pts = [(np.random.randint(0, 255), np.random.randint(0, 319)) for i in range(500)]
    raw_sampled_sum = 0
    pred_sample_sum = 0
    for pt in random_pts:
        if not img_origin[pt[0], pt[1]]:
            continue
        raw_sampled_sum += img_origin[pt[0], pt[1]]
        pred_sample_sum += pred[0, pt[0], pt[1]]
    depth_ratio = raw_sampled_sum/pred_sample_sum
    pred *= depth_ratio

    vmin = pred.min()
    vmax = pred.max()

    print("polars", vmax, vmin, nmax, nmin)


    axes[0, 0].set_title("RGB")
    axes[0, 0].imshow((color - color.min()) / (color.max() - color.min()))

    axes[0, 1].set_title("raw_depth")
    img = axes[0, 1].imshow(raw_depth[0], cmap="RdBu_r")
    fig.colorbar(img, ax=axes[0, 1])

    axes[1, 0].set_title("mask")
    axes[1, 0].imshow(mask[0])

    axes[1, 1].set_title("pred")
    img = axes[1, 1].imshow(pred[0], cmap="RdBu_r", vmin=nmin, vmax=nmax)
    fig.colorbar(img, ax=axes[1, 1])
    upsampled_image = cv.resize(pred[0].numpy().astype('uint16'), None, fx=1920/320, fy=1080/256, interpolation=cv.INTER_LINEAR)
    cv.imwrite("/root/saic_depth_completion/output/pred_{}.png".format(saveidx), upsampled_image)
    print("shape of the upsampled image is ", upsampled_image.shape)

    axes[2, 0].set_title("spiral search")
    img = axes[2, 0].imshow(img_spiral, cmap="RdBu_r", vmin=nmin, vmax=nmax)
    fig.colorbar(img, ax=axes[2, 0])
    upsampled_image = cv.resize(img_spiral.astype('uint16'), None, fx=1920/320, fy=1080/256, interpolation=cv.INTER_LINEAR)
    cv.imwrite("/root/saic_depth_completion/output/spiral.png", upsampled_image)

    axes[2, 1].set_title("average search")
    img = axes[2, 1].imshow(img_average, cmap="RdBu_r", vmin=nmin, vmax=nmax)
    fig.colorbar(img, ax=axes[2, 1])
    upsampled_image = cv.resize(img_average.astype('uint16'), None, fx=1920/320, fy=1080/256, interpolation=cv.INTER_LINEAR)
    cv.imwrite("/root/saic_depth_completion/output/average.png", upsampled_image)

    if close:
        plt.close(fig)
    return fig
