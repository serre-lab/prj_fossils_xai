from typing import Dict, Union
import os
import pandas as pd
import numpy as np
import tensorflow as tf


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


def symmetric_batch(batch_x):
    return tf.concat([augmentations(batch_x), augmentations(batch_x)], axis=0)


def resize_images(batch_x, width=224, height=224):
    return tf.image.resize(batch_x, (width, height))


def data_config(username="irodri15_oscar"):
    if username == "irodri15_oscar":
        local_datasets_dir = "/users/irodri15/data/irodri15/Fossils/Experiments/softmax_triplet/datasets/"
        pretrained_weights_dir = "/users/irodri15/data/irodri15/Fossils/Experiments/softmax_triplet/pretrained/"
        training_models_dir = "e"
        caffe_iter_size = 1
        logging_threshold = 100
        batch_size = 32
    else:
        print("not implemented")
    return local_datasets_dir, pretrained_weights_dir, training_models_dir


def load_train_test_val(train_csv_file, test_csv_file, val_csv_file):
    train_df, test_df, val_df = (
        pd.read_csv(train_csv_file),
        pd.read_csv(test_csv_file),
        pd.read_csv(val_csv_file),
    )
    # import pdb;pdb.set_trace()
    # make domains to have same amount of data.
    try:
        labels, counts = np.unique(train_df.domain.tolist(), return_counts=True)
        ratio = np.max(counts) // np.min(counts)
        under_rep_domain = labels[np.argmin(counts)]
        train_df = pd.concat(
            [train_df] + [train_df[train_df.domain == under_rep_domain]] * ratio
        )
    except:
        print("domain not present")
    return {"train": train_df, "val": val_df, "test": test_df}


def load_dataset_from_file(
    username="irodri15_oscar",
    dataname="leaves_fossils_fewshot_v1.0",
    split=None,
    thresh=None,
    label_out=None,
    shot=None,
):
    csv_path = "./csv/fossils.csv"
    test_df = pd.read_csv(csv_path)

    return {"test": test_df}


def load_data_from_tensor_slices(
    data: pd.DataFrame,
    training=False,
    seed=42,
    x_col="path",
    y_col="label",
    d_col="domain",
    dtype=tf.float32,
    number_classes=19,
    wsize=600,
    hsize=600,
    gray=False,
):
    dtype = dtype or tf.uint8
    num_samples = data.shape[0]

    def load_img(image_path, gray=False):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        if gray:
            img = tf.image.rgb_to_grayscale(img)
            img = tf.image.grayscale_to_rgb(img)
        img = tf.image.resize(img, (wsize, hsize))
        return img

    # print("\n\n\n\n columns are : ", list(data[d_col].columns), "\n\n\n")
    x_data = tf.data.Dataset.from_tensor_slices(data[x_col].values.tolist())
    y_data = tf.data.Dataset.from_tensor_slices(
        data[y_col].astype("int").values.tolist()
    )
    d_data = tf.data.Dataset.from_tensor_slices(
        data[d_col].astype("int").values.tolist()
    )

    data = tf.data.Dataset.zip((x_data, y_data, d_data))
    data = data.map(lambda x, y, d: {"x": x, "y": y, "d": d})
    data = data.take(num_samples).cache()

    # TODO TEST performance and randomness of the order of shuffle and cache when shuffling full dataset each iteration, but only filepaths and not full images.
    if training:
        data = data.shuffle(num_samples, seed=seed, reshuffle_each_iteration=True)

    # import pdb;pdb.set_trace()
    data = data.map(
        lambda example: {
            "x": tf.image.convert_image_dtype(
                load_img(example["x"], gray=gray), dtype=dtype
            ),
            "y": tf.one_hot(example["y"], number_classes),
            "d": example["d"],
            "p": example["x"],
        },
        num_parallel_calls=-1,
    )

    return data


def extract_data(
    data,
    number_classes=19,
    hsize=600,
    wsize=600,
    shuffle_first=True,
    seed=None,
    gray=False,
):
    subset_keys = list(data.keys())

    extracted_data = {}
    for subset in subset_keys:
        if shuffle_first:
            data[subset] = data[subset].sample(frac=1)

        paths = data[subset]["file_name"]
        labels = data[subset]["label"]
        if "domain" not in data[subset].columns:
            data[subset]["domain"] = [1] * len(data[subset])
        domains = data[subset]["domain"]

        extracted_data[subset] = pd.DataFrame.from_records(
            [
                {"path": path, "label": label, "domain": text_label}
                for path, label, text_label in zip(paths, labels, domains)
            ]
        )

        training = subset == "train"
        extracted_data[subset] = load_data_from_tensor_slices(
            data=extracted_data[subset],
            number_classes=number_classes,
            training=training,
            seed=seed,
            x_col="path",
            y_col="label",
            wsize=wsize,
            hsize=hsize,
            dtype=tf.float32,
            gray=gray,
        )

    return extracted_data


def load_and_extract_data(
    number_classes=19,
    username="irodri15_oscar",
    dataname="leaves_fossils_fewshot_v1.0",
    hsize=600,
    wsize=600,
    label_out=None,
    split=None,
    shot=None,
    shuffle_first=True,
    seed=None,
    batch_size=32,
    gray=False,
    thresh=None,
    computing_viz=False,
):
    data = load_dataset_from_file(
        username="irodri15_oscar",
        dataname=dataname,
        label_out=label_out,
        shot=shot,
        split=split,
        thresh=thresh,
    )

    data = extract_data(
        data,
        shuffle_first=shuffle_first,
        hsize=hsize,
        wsize=wsize,
        seed=seed,
        number_classes=number_classes,
        gray=gray,
    )

    test_dataset = data["test"].batch(batch_size)

    return test_dataset


def leaves_fewshot_ds(
    label_out,
    shot,
    split=None,
    wsize=600,
    hsize=600,
    batch_size=32,
    dataname="leaves_fossils_fewshot_v1.0",
    number_classes=19,
    gray=False,
    thresh=None,
    computing_viz=False,
):
    test = load_and_extract_data(
        label_out=label_out,
        wsize=wsize,
        hsize=hsize,
        shot=shot,
        split=split,
        number_classes=number_classes,
        batch_size=batch_size,
        dataname=dataname,
        gray=gray,
        thresh=thresh,
        computing_viz=computing_viz,
    )

    if computing_viz:
        test = test.map(
            lambda sample: (
                resize_images(sample["x"], wsize, hsize),
                sample["y"],
                sample["d"],
                sample["p"],
            )
        )
    else:
        test = test.map(
            lambda sample: (
                resize_images(sample["x"], wsize, hsize),
                sample["y"],
                sample["d"],
            )
        )

    return test


def mocking_ds():
    """What I need"""
    nb_samples = 200
    size = 600

    inputs = tf.random.normal((nb_samples, size, size, 3))  # (N, 600, 600, 3)
    labels = tf.one_hot(
        tf.argmax(tf.random.normal((nb_samples, 19)), -1), 19
    )  # (N, 19) one hot encoded classes
    domains = tf.cast(
        tf.random.normal((nb_samples,)) > 0.5, tf.int32
    )  # (N,) int32 1 or zero

    train_ds = (
        tf.data.Dataset.from_tensor_slices((inputs, labels, domains)).batch(32).repeat()
    )
    test_ds = tf.data.Dataset.from_tensor_slices((inputs, labels, domains)).batch(32)

    return train_ds, test_ds
