# import keras_cv
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

import torch

# torch.cuda.set_per_process_memory_fraction(0.2, device=0)

import torch.nn.functional as F
import os
import numpy as np

# num_classes = len(class_names)
# AUTO = tf.data.AUTOTUNE
# rand_augment = keras_cv.layers.RandAugment(value_range = (-1, 1), augmentations_per_image = 3, magnitude=0.5)


def get_fossils_paths(class_names, class_to_id, id_to_class, data_path):
    base = data_path

    x_fossils = {}
    dataset = []

    total_images = 0

    for cls in class_names:
        if not os.path.exists(base + "/" + cls):
            continue
        paths = [base + "/" + cls + "/" + p for p in os.listdir(base + "/" + cls)]
        if len(paths) > 0:
            dataset = [(p, str(class_to_id[cls])) for p in paths]
            x_fossils[int(class_to_id[cls])] = dataset
        else:
            print(cls)

        dataset = np.array(dataset)
        # print(f'len of classes : {len(class_names)}')

    total_images = 0
    for key, val in x_fossils.items():
        print(f"{id_to_class[key]}({key}) : {len(val)}")
        total_images += len(val)

    print(f"set size : {total_images}")

    return x_fossils


SIZE = 384
debug = None
wsize = hsize = SIZE


def augmentations(
    x, crop_size=22, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
):
    x = tf.cast(x, tf.float32)
    x = tf.image.random_crop(x, (tf.shape(x)[0], 100, 100, 3))
    x = tf.image.random_brightness(x, max_delta=brightness)
    x = tf.image.random_contrast(x, lower=1.0 - contrast, upper=1 + contrast)
    x = tf.image.random_saturation(x, lower=1.0 - saturation, upper=1.0 + saturation)
    x = tf.image.random_hue(x, max_delta=hue)
    x = tf.image.resize(x, (128, 128))
    x = tf.clip_by_value(x, 0.0, 255.0)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    return x


def pad_gt(x, sam):
    h, w = x.shape[-2:]
    padh = sam.image_encoder.img_size - h
    padw = sam.image_encoder.img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def preprocess(img, predictor, sam):
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


def parse_fossils(element, num_classes, randaugment, maskaugment=True):
    # global debug
    path, class_id = element[0], element[1]
    img = load_img(path)

    # data_mask = tf.io.read_file(path_mask)
    # mask = tf.io.decode_jpeg(data_mask)

    class_id = tf.strings.to_number(class_id)
    class_id = tf.cast(class_id, tf.int32)

    label = tf.one_hot(class_id, num_classes)

    # img = pi(img, mask)
    # img = tf.image.resize_with_pad(img, SIZE, SIZE, method="bicubic", antialias=True)

    return tf.cast(img, tf.float32), tf.cast(label, tf.int32)


def resize_images(batch_x, width=224, height=224):
    return tf.image.resize(batch_x, (width, height))


def load_img(image_path, gray=False):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if gray:
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize(img, (wsize, hsize))
    return img
