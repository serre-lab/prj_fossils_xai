import os

# import keras_cv
import tensorflow as tf
import numpy as np


def get_leaves_paths(class_names, class_to_id, id_to_class, data_path):
    base = data_path

    x_leaves = {}

    for cls in class_names:
        class_path = os.path.join(base, cls)
        if not os.path.exists(class_path):
            continue

        paths = [base + "/" + cls + "/" + p for p in os.listdir(class_path)]
        try:
            dataset = [(p, str(class_to_id[cls])) for p in paths]
            x_leaves[int(class_to_id[cls])] = dataset
        except:
            print(cls)

    total_images = 0
    for key, val in x_leaves.items():
        print(f"Leaves {id_to_class[key]} ({key}) : {len(val)}")
        total_images += len(val)

    print(f"Test set of Leaves : {total_images}")
    return x_leaves


# # num_classes = len(class_names)
# AUTO = tf.data.AUTOTUNE
# rand_augment = keras_cv.layers.RandAugment(value_range = (-1, 1), augmentations_per_image = 3, magnitude=0.5)

SIZE = 384
debug = None


def normalize(img):
    img = img - tf.math.reduce_min(img)
    img = img / tf.math.reduce_max(img)
    img = img * 2.0 - 1.0
    return img


def smooth_mask(mask, ds=20):
    shape = tf.shape(mask)
    w, h = shape[0], shape[1]
    return tf.image.resize(
        tf.image.resize(mask, (ds, ds), method="bicubic"), (w, h), method="bicubic"
    )


def resize(img):
    # default resize function for all pi outputs
    return tf.image.resize(img, (SIZE, SIZE), method="bicubic")


# def pi(img, mask):
#   img = tf.cast(img, tf.float32)
#   shape = tf.shape(img)
#   w, h = tf.cast(shape[0], tf.int64), tf.cast(shape[1], tf.int64)
#   mask = smooth_mask(mask.cpu().numpy().astype(float))
#   mask = tf.reduce_mean(mask, -1)
#   img = img * tf.cast(mask > 0.01, tf.float32)[:, :, None]
#   img_resize = tf.image.resize(img, (SIZE, SIZE), method="bicubic", antialias=True)
#   img_pad = tf.image.resize_with_pad(img, SIZE, SIZE, method="bicubic", antialias=True)
#   # building 2 anchors
#   anchors = tf.where(mask > 0.15)
#   anchor_xmin = tf.math.reduce_min(anchors[:, 0])
#   anchor_xmax = tf.math.reduce_max(anchors[:, 0])
#   anchor_ymin = tf.math.reduce_min(anchors[:, 1])
#   anchor_ymax = tf.math.reduce_max(anchors[:, 1])
#   if anchor_xmax - anchor_xmin > 50 and anchor_ymax - anchor_ymin > 50:
#     img_anchor_1 = resize(img[anchor_xmin:anchor_xmax, anchor_ymin:anchor_ymax])
#     delta_x = (anchor_xmax - anchor_xmin) // 4
#     delta_y = (anchor_ymax - anchor_ymin) // 4
#     img_anchor_2 = img[anchor_xmin+delta_x:anchor_xmax-delta_x,
#                       anchor_ymin+delta_y:anchor_ymax-delta_y]
#     img_anchor_2 = resize(img_anchor_2)
#   else:
#     img_anchor_1 = img_resize
#     img_anchor_2 = img_pad
#   # building the anchors max
#   anchor_max = tf.where(mask == tf.math.reduce_max(mask))[0]
#   anchor_max_x, anchor_max_y = anchor_max[0], anchor_max[1]
#   img_max_zoom1 = img[tf.math.maximum(anchor_max_x-SIZE, 0): tf.math.minimum(anchor_max_x+SIZE, w),
#                       tf.math.maximum(anchor_max_y-SIZE, 0): tf.math.minimum(anchor_max_y+SIZE, h)]
#   img_max_zoom1 = resize(img_max_zoom1)
#   img_max_zoom2 = img[anchor_max_x-SIZE//2:anchor_max_x+SIZE//2,
#                       anchor_max_y-SIZE//2:anchor_max_y+SIZE//2]
#   img_max_zoom2 = img[tf.math.maximum(anchor_max_x-SIZE//2, 0): tf.math.minimum(anchor_max_x+SIZE//2, w),
#                       tf.math.maximum(anchor_max_y-SIZE//2, 0): tf.math.minimum(anchor_max_y+SIZE//2, h)]
#   #tf.print(img_max_zoom2.shape)
#   #img_max_zoom2 = resize(img_max_zoom2)
#   return tf.cast([
#       img_resize,
#       #img_pad,
#       img_anchor_1,
#       img_anchor_2,
#       img_max_zoom1,
#       #img_max_zoom2,
#     ], tf.float32)


def pi(img, mask):
    img = tf.cast(img, tf.float32)
    shape = tf.shape(img)
    w, h = tf.cast(shape[0], tf.int64), tf.cast(shape[1], tf.int64)
    mask = smooth_mask(mask.cpu().numpy().astype(float))
    # mask = tf.reduce_mean(mask, -1)
    img = img * tf.cast(mask[0] > 0.01, tf.float32)[:, :, None]
    img_resize = tf.image.resize(img, (SIZE, SIZE), method="bicubic", antialias=True)
    img_pad = tf.image.resize_with_pad(
        img, SIZE, SIZE, method="bicubic", antialias=True
    )
    # building 2 anchors
    anchors = tf.where(mask > 0.15)
    anchor_xmin = tf.math.reduce_min(anchors[:, 0])
    anchor_xmax = tf.math.reduce_max(anchors[:, 0])
    anchor_ymin = tf.math.reduce_min(anchors[:, 1])
    anchor_ymax = tf.math.reduce_max(anchors[:, 1])
    if anchor_xmax - anchor_xmin > 50 and anchor_ymax - anchor_ymin > 50:
        img_anchor_1 = resize(img[anchor_xmin:anchor_xmax, anchor_ymin:anchor_ymax])
        delta_x = (anchor_xmax - anchor_xmin) // 4
        delta_y = (anchor_ymax - anchor_ymin) // 4
        img_anchor_2 = img[
            anchor_xmin + delta_x : anchor_xmax - delta_x,
            anchor_ymin + delta_y : anchor_ymax - delta_y,
        ]
        img_anchor_2 = resize(img_anchor_2)
    else:
        img_anchor_1 = img_resize
        img_anchor_2 = img_pad
    # building the anchors max
    anchor_max = tf.where(mask == tf.math.reduce_max(mask))[0]
    anchor_max_x, anchor_max_y = anchor_max[0], anchor_max[1]
    img_max_zoom1 = img[
        tf.math.maximum(anchor_max_x - SIZE, 0) : tf.math.minimum(
            anchor_max_x + SIZE, w
        ),
        tf.math.maximum(anchor_max_y - SIZE, 0) : tf.math.minimum(
            anchor_max_y + SIZE, h
        ),
    ]
    img_max_zoom1 = resize(img_max_zoom1)
    img_max_zoom2 = img[
        anchor_max_x - SIZE // 2 : anchor_max_x + SIZE // 2,
        anchor_max_y - SIZE // 2 : anchor_max_y + SIZE // 2,
    ]
    img_max_zoom2 = img[
        tf.math.maximum(anchor_max_x - SIZE // 2, 0) : tf.math.minimum(
            anchor_max_x + SIZE // 2, w
        ),
        tf.math.maximum(anchor_max_y - SIZE // 2, 0) : tf.math.minimum(
            anchor_max_y + SIZE // 2, h
        ),
    ]
    # tf.print(img_max_zoom2.shape)
    # img_max_zoom2 = resize(img_max_zoom2)
    return tf.cast(
        [
            img_resize,
            # img_pad,
            img_anchor_1,
            img_anchor_2,
            img_max_zoom1,
            # img_max_zoom2,
        ],
        tf.float32,
    )


# def pi(img, mask):
#   # print(img.shape, type(img), mask.shape, type(mask))
#   img = tf.cast(img, tf.float32)
#   shape = tf.shape(img)
#   w, h = tf.cast(shape[0], tf.int64), tf.cast(shape[1], tf.int64)
#   mask = smooth_mask(mask)
#   mask = tf.reduce_mean(mask, -1)

#   img = img * tf.cast(mask > 0.1, tf.float32)[:, :, None]

#   img_resize = tf.image.resize(img, (SIZE, SIZE), method="bicubic", antialias=True)
#   img_pad = tf.image.resize_with_pad(img, SIZE, SIZE, method="bicubic", antialias=True)

#   # building 2 anchors
#   anchors = tf.where(mask > 0.15)
#   anchor_xmin = tf.math.reduce_min(anchors[:, 0])
#   anchor_xmax = tf.math.reduce_max(anchors[:, 0])
#   anchor_ymin = tf.math.reduce_min(anchors[:, 1])
#   anchor_ymax = tf.math.reduce_max(anchors[:, 1])

#   if anchor_xmax - anchor_xmin > 50 and anchor_ymax - anchor_ymin > 50:

#     img_anchor_1 = resize(img[anchor_xmin:anchor_xmax, anchor_ymin:anchor_ymax])

#     delta_x = (anchor_xmax - anchor_xmin) // 4
#     delta_y = (anchor_ymax - anchor_ymin) // 4
#     img_anchor_2 = img[anchor_xmin+delta_x:anchor_xmax-delta_x,
#                       anchor_ymin+delta_y:anchor_ymax-delta_y]
#     img_anchor_2 = resize(img_anchor_2)
#   else:
#     img_anchor_1 = img_resize
#     img_anchor_2 = img_pad

#   # building the anchors max
#   anchor_max = tf.where(mask == tf.math.reduce_max(mask))[0]
#   anchor_max_x, anchor_max_y = anchor_max[0], anchor_max[1]

#   img_max_zoom1 = img[tf.math.maximum(anchor_max_x-SIZE, 0): tf.math.minimum(anchor_max_x+SIZE, w),
#                       tf.math.maximum(anchor_max_y-SIZE, 0): tf.math.minimum(anchor_max_y+SIZE, h)]

#   img_max_zoom1 = resize(img_max_zoom1)
#   img_max_zoom2 = img[anchor_max_x-SIZE//2:anchor_max_x+SIZE//2,
#                       anchor_max_y-SIZE//2:anchor_max_y+SIZE//2]
#   img_max_zoom2 = img[tf.math.maximum(anchor_max_x-SIZE//2, 0): tf.math.minimum(anchor_max_x+SIZE//2, w),
#                       tf.math.maximum(anchor_max_y-SIZE//2, 0): tf.math.minimum(anchor_max_y+SIZE//2, h)]
#   #tf.print(img_max_zoom2.shape)
#   # img_max_zoom2 = resize(img_max_zoom2)

#   return tf.cast(img_resize, tf.float32), tf.cast(img_max_zoom1, tf.float32)


def parse_leaves(element, num_classes, randaugment, maskaugment=True):
    # global debug
    path, path_mask, class_id = element[0], element[1], element[2]

    data = tf.io.read_file(path)
    # data = tf.cast(data, dtype = tf.float32)
    img = tf.io.decode_jpeg(data)
    img = tf.cast(img, tf.uint8)
    img = normalize(img)
    shape = tf.shape(img)

    data_mask = tf.io.read_file(path_mask)
    mask = tf.io.decode_jpeg(data_mask)

    class_id = tf.strings.to_number(class_id)
    class_id = tf.cast(class_id, tf.int32)

    label = tf.one_hot(class_id, num_classes)

    img, img_zoom = pi(img, mask)
    img = tf.image.resize_with_pad(img, SIZE, SIZE, method="bicubic", antialias=True)
    img_zoom = tf.image.resize_with_pad(
        img_zoom, SIZE, SIZE, method="bicubic", antialias=True
    )
    # img_zoom2 = tf.image.resize_with_pad(img_zoom2, SIZE, SIZE, method="bicubic", antialias=True)

    return (
        tf.cast(img, tf.float32),
        tf.cast(img_zoom, tf.float32),
        tf.cast(label, tf.int32),
    )


def load_leaves(cid, x_tests):
    dataset = x_tests[cid]
    images = []

    labels = []
    domain = []

    for i, ele in enumerate(dataset):
        image_path, class_id = ele
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (SIZE, SIZE))
        img = tf.image.convert_image_dtype(img, tf.float32)

        images.append(img)
        labels.append(class_id)
        domain.append(0)

    return images, labels, domain
