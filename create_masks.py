import torch

# torch.cuda.set_per_process_memory_fraction(0.2, device=0)
print("segment anything")
from segment_anything import SamPredictor, sam_model_registry

print("importing")
sam = sam_model_registry["default"]("./models/sam_02-06_dice.pth")
sam.cuda()
print("defining")
predictor = SamPredictor(sam)

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
import os
import numpy as np

# from losses import *
# from models import get_triplet_model,get_triplet_model_simclr,get_triplet_model_beit
from data import mocking_ds, leaves_fewshot_ds
from sklearn.metrics import classification_report
from Craft.craft.new_craft_tf import Craft
import pandas as pd
import multiprocessing as mp

# import segmentation_models as sm
import json

# sm.set_framework('tf.keras')
# sm.framework()
import cv2

import helpers

print("Setting Parameters")
AUTOTUNE = tf.data.AUTOTUNE
MARGIN = 0.152
EPOCHS = 50
LR = 0.006515
LAMBDA_TRIPLET_CLASS = 0.343 * 2
LAMBDA_TRIPLET_XDOMAIN = 0.343
NUMBER_CLASSES = 55
CKPT_DIRECTORY = (
    "/users/irodri15/data/irodri15/Fossils/Experiments/softmax_triplet_tf2.0"
)
NAME = "TEST_beit"
SIZE = 384

CrossEntropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

from torch.nn import functional as F


def show(img, p=False, smooth=False, **kwargs):
    """Display torch/tf tensor"""
    try:
        img = img.detach().cpu()
    except:
        img = np.array(img)

    img = np.array(img, dtype=np.float32)

    # check if channel first
    if img.shape[0] == 1:
        img = img[0]
    elif img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
    # check if cmap
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    # normalize
    if img.max() > 1 or img.min() < 0:
        img -= img.min()
        img /= img.max()
    # check if clip percentile
    if p is not False:
        img = np.clip(img, np.percentile(img, p), np.percentile(img, 100 - p))

    if smooth and len(img.shape) == 2:
        img = gaussian_filter(img, smooth)

    plt.imshow(img, **kwargs)
    plt.axis("off")
    plt.grid(None)


def pad_gt(x):
    h, w = x.shape[-2:]
    padh = sam.image_encoder.img_size - h
    padw = sam.image_encoder.img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def preprocess(img):
    img = np.array(img).astype(np.uint8)

    # assert img.max() > 127.0

    img_preprocess = predictor.transform.apply_image(img)
    intermediate_shape = img_preprocess.shape

    img_preprocess = torch.as_tensor(img_preprocess).cuda()
    img_preprocess = img_preprocess.permute(2, 0, 1).contiguous()[None, :, :, :]

    img_preprocess = sam.preprocess(img_preprocess)
    if len(intermediate_shape) == 3:
        intermediate_shape = intermediate_shape[:2]
    elif len(intermediate_shape) == 4:
        intermediate_shape = intermediate_shape[1:3]

    return img_preprocess, intermediate_shape


def normalize(img):
    img = img - tf.math.reduce_min(img)
    img = img / tf.math.reduce_max(img)
    img = img * 2.0 - 1.0
    return img


def resize(img):
    # default resize function for all pi outputs
    return tf.image.resize(img, (SIZE, SIZE), method="bicubic")


def smooth_mask(mask, ds=7):
    shape = tf.shape(mask)
    w, h = shape[0], shape[1]
    ## apply a gaussian filter to the mask
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, -1)
    mask = tf.image.resize(mask, (ds, ds), method="bicubic")
    mask = tf.image.resize(mask, (w, h), method="bicubic")
    mask = tf.squeeze(mask, -1)
    return mask


def gaussian_kernel(kernel_size, sigma):
    """Manually creates a Gaussian kernel."""
    x_range = tf.range(-(kernel_size // 2), kernel_size // 2 + 1, dtype=tf.float32)
    y_range = tf.range(-(kernel_size // 2), kernel_size // 2 + 1, dtype=tf.float32)
    x, y = tf.meshgrid(x_range, y_range, indexing="ij")
    gaussian_kernel = tf.exp(-(tf.square(x) + tf.square(y)) / (2 * tf.square(sigma)))
    return gaussian_kernel / tf.reduce_sum(gaussian_kernel)


def smooth_mask_v2(mask, kernel_size=5, sigma=1.0):
    """Applies Gaussian smoothing on a mask."""

    # Add batch and channel dimensions
    mask = mask[tf.newaxis, ..., tf.newaxis]
    # Create Gaussian kernel
    gauss_kernel = gaussian_kernel(kernel_size, sigma)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    # Apply Gaussian filter
    smoothed_mask = tf.nn.conv2d(
        mask, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME"
    )
    # Remove batch and channel dimensions
    smoothed_mask = tf.squeeze(smoothed_mask)
    return smoothed_mask


def one_step_inference(x):
    if len(x.shape) == 3:
        original_size = x.shape[:2]
    elif len(x.shape) == 4:
        original_size = x.shape[1:3]

    x, intermediate_shape = preprocess(x)

    with torch.no_grad():
        image_embedding = sam.image_encoder(x)

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        if len(x.shape) == 3:
            input_size = tuple(x.shape[:2])
        elif len(x.shape) == 4:
            input_size = tuple(x.shape[-2:])

        # upscaled_masks = sam.postprocess_masks(low_res_masks, input_size, original_size).cuda()
        mask = F.interpolate(low_res_masks, (1024, 1024))[
            :, :, : intermediate_shape[0], : intermediate_shape[1]
        ]
        mask = F.interpolate(mask, (original_size[0], original_size[1]))

    return mask


def segmentation_sam(batch_input, batch_labels, batch_domain, batch_size):
    X = tf.image.resize_with_pad(batch_input, SIZE, SIZE)
    samples = []
    labels = []
    domains = []
    un_mask_samples = []
    for x, y, d in zip(X, batch_labels, batch_domain):
        predicted_mask = one_step_inference(x)
        # X = seg_preprocess_input(batch_input).numpy()
        # out = seg_model.predict(X, batch_size=batch_size)

        mask = predicted_mask > 0.99
        mask = mask[0]
        total_mask = mask.shape[0] * mask.shape[1]
        mask_sum = mask.sum()
        mask = mask.cpu()
        # without mask
        if d == 1:
            # if mask_sum <total_mask*0.9:
            mask = torch.ones_like(mask)
        else:
            if mask_sum < total_mask * 0.15:
                mask = torch.ones_like(mask)
        import ipdb

        ipdb.set_trace()
        mask = mask.numpy()[0] * 255
        cv2.imwrite("mask_path.png", mask)
    return


def evaluate(
    leaves_data_dir, leaves_masks, batch_size=10, prob_augmentation=0.9, val_ds=None
):
    all_classes = os.listdir(leaves_data_dir)

    for cls in all_classes:
        print(f"current class: {cls}")
        class_path = os.path.join(leaves_data_dir, cls)
        leaves_masks_path = os.path.join(leaves_masks, cls)
        os.makedirs(leaves_masks_path, exist_ok=True)
        images_names = os.listdir(class_path)
        for img in images_names:
            image_path = os.path.join(class_path, img)
            image = cv2.imread(image_path)
            # image = cv2.resize(image, (384,384))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(image)
            predicted_mask = one_step_inference(x)
            mask = predicted_mask > 0.99
            mask = mask[0]
            total_mask = mask.shape[0] * mask.shape[1]
            mask_sum = mask.sum()
            mask = mask.cpu()
            # without mask

            if mask_sum < total_mask * 0.15:
                mask = torch.ones_like(mask)
            mask = mask.numpy()[0] * 255
            mask_path = os.path.join(leaves_masks_path, img)
            cv2.imwrite(mask_path, mask)


if __name__ == "__main__":
    csv_path = "./csv/fossils.csv"
    fossils_data_dir = (
        "/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Florissant_Fossil_v2.0"
    )
    leaves_data_dir = (
        "/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Extant_Leaves"
    )
    leaves_masks = "./mask_images_all_leaves2/"

    # model, g, h = helpers.get_model(model_path)
    batch_size = 64
    SIZE = 384

    print("dataset loaded")
    evaluate(leaves_data_dir, leaves_masks, batch_size=batch_size)
