import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from math import ceil
from sklearn.decomposition import NMF
import cv2
import xplique
from xplique.features_visualizations import Objective
from xplique.features_visualizations import maco

# from xplique.plot import plot_maco
from tqdm import tqdm
import shutil

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

fossils_dir = (
    "/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Florissant_Fossil_v2.0"
)
leaves_dir = "/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Extant_Leaves"
plot_save_dir = (
    "/cifs/data/tserre_lrs/projects/prj_fossils_xai/concepts/fossils_concept"
)
plot_leaves_save_dir = (
    "/cifs/data/tserre_lrs/projects/prj_fossils_xai/concepts/leaves_concept"
)
save_feature_viz = "/cifs/data/tserre_lrs/projects/prj_fossils_xai/maco/fossils_viz"
save_leaves_feature_viz = (
    "/cifs/data/tserre_lrs/projects/prj_fossils_xai/maco/leaves_viz"
)
mask_dir = "/cifs/data/tserre_lrs/projects/prj_fossils_xai/mask_images_all_leaves2"
model_path = "/cifs/data/tserre_lrs/projects/prj_fossils_xai/new_models/model-14_RESNET_101_TRIPLET.h5"

classes = [
    "Anacardiaceae",
    "Berberidaceae",
    "Betulaceae",
    "Cupressaceae",
    "Dryopteridaceae",
    "Fabaceae",
    "Fagaceae",
    "Juglandaceae",
    "Lauraceae",
    "Meliaceae",
    "Myrtaceae",
    "Pinaceae",
    "Rhamnaceae",
    "Rosaceae",
    "Salicaceae",
    "Sapindaceae",
]


def disk_usage(dir, classes):
    total_images = 0
    for cls in classes:
        class_dir = os.path.join(dir, cls)
        total_images += len(os.listdir(class_dir))
    print(total_images)


disk_usage(leaves_dir, classes)
import ipdb

ipdb.set_trace()

cce = tf.keras.losses.categorical_crossentropy
model = tf.keras.models.load_model(model_path, custom_objects={"cce": cce})
print(model.summary())
features = tf.keras.Model(model.input, model.layers[-5].output)


def load_fossils_dir(class_name, fossils_dir):
    class_dir = os.path.join(fossils_dir, class_name)
    paths = os.listdir(class_dir)
    fossils = []
    count = 0
    for p in paths:
        fossils_path = os.path.join(class_dir, p)
        img = cv2.imread(fossils_path)[..., ::-1]
        img = img.astype(np.float32)
        fossils.append(img)
        count += 1
    print(f"total images: {count}")
    return fossils


def load_leaves_dir(class_name, leaves_dir, mask_dir):
    mask_dir = os.path.join(mask_dir, class_name)
    leaves_dir = os.path.join(leaves_dir, class_name)
    paths = os.listdir(mask_dir)
    masked_imgs = []
    imgs = []
    count = 0
    for p in paths[:100]:
        mask_path = os.path.join(mask_dir, p)
        mask = cv2.imread(mask_path) / 225.0
        img = cv2.imread(image_dir + "/" + p)[..., ::-1]
        img = img.astype(np.float32)
        image = img * (mask > 0.1).astype(np.float32)
        masked_imgs.append(image)
        imgs.append(img)
        count += 1
    print(f"total images: {count}")
    return imgs, masked_imgs


def generate_square_crops(image, crop_size=1000):
    height, width, _ = image.shape
    crops = []
    y_steps = ceil(height / crop_size)
    x_steps = ceil(width / crop_size)
    threshold = 0.9
    for y in range(y_steps):
        for x in range(x_steps):
            start_y = y * crop_size
            end_y = min(start_y + crop_size, height)
            start_x = x * crop_size
            end_x = min(start_x + crop_size, width)

            # If we are at the end, take more from the other side
            if end_y - start_y < crop_size:
                start_y = max(0, end_y - crop_size)
            if end_x - start_x < crop_size:
                start_x = max(0, end_x - crop_size)

            crop = image[start_y:end_y, start_x:end_x, :]
            crops.append(crop)
    return np.array(crops)


def preprocess(x):
    return x / 255.0


for i in range(len(classes)):
    class_id = 0
    nb_concepts = 40
    CROPS = []
    ACTIVATIONS = []

    class_save_dir_c = os.path.join(plot_save_dir, classes[class_id], "coalesce")
    class_save_dir_ind = os.path.join(plot_save_dir, classes[class_id], "individual")
    class_viz_dir = os.path.join(save_feature_viz, classes[class_id])
    os.makedirs(class_save_dir_c, exist_ok=True)
    os.makedirs(class_save_dir_ind, exist_ok=True)
    os.makedirs(class_viz_dir, exist_ok=True)

    print(f"Fossils Dir: {fossils_dir}")
    print(f"Plot Dir: {plot_save_dir}")
    print(f"Class Save Dir: {class_save_dir_c}")
    print(f"class save dir ind: {class_save_dir_ind}")
    print(f"ViZ dir: {class_viz_dir}")

    X = load_fossils_dir(classes[class_id], fossils_dir)

    count = 0
    for i, x in enumerate(X):
        crops = generate_square_crops(x)
        crops = tf.image.resize(crops, (384, 384))
        CROPS += list(crops.numpy().astype(np.uint8))
        crops = preprocess(crops)
        activations = features(crops)
        ACTIVATIONS += list(activations.numpy())
        count += 1

    ACTIVATIONS = np.array(ACTIVATIONS)
    CROPS = np.array(CROPS)
    print(ACTIVATIONS.shape)
    np.save("activations2/activations.npy", ACTIVATIONS)
