import torch

torch.cuda.set_per_process_memory_fraction(0.2, device=0)
print("segment anything")
from segment_anything import SamPredictor, sam_model_registry

print("importing")
sam = sam_model_registry["default"]("sam_02-06_dice_0.pth")
sam.cuda()
print("defining")
predictor = SamPredictor(sam)

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
import os
import numpy as np
from losses import *
from models import get_triplet_model, get_triplet_model_simclr, get_triplet_model_beit
from data import mocking_ds, leaves_fewshot_ds
from sklearn.metrics import classification_report

import pandas as pd
import multiprocessing as mp

# import segmentation_models as sm
import json

# sm.set_framework('tf.keras')
# sm.framework()


print("Setting Parameters")
AUTOTUNE = tf.data.AUTOTUNE
MARGIN = 0.152
EPOCHS = 50
LR = 0.006515
LAMBDA_TRIPLET_CLASS = 0.343
LAMBDA_TRIPLET_XDOMAIN = 0.343
NUMBER_CLASSES = 55
CKPT_DIRECTORY = (
    "/users/irodri15/data/irodri15/Fossils/Experiments/softmax_triplet_tf2.0"
)
NAME = "TEST_beit"
SIZE = 384

CrossEntropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


from torch.nn import functional as F


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


def smooth_mask(mask, ds=20):
    shape = tf.shape(mask)
    w, h = shape[0], shape[1]
    return tf.image.resize(
        tf.image.resize(mask, (ds, ds), method="bicubic"), (w, h), method="bicubic"
    )


def pi(img, mask):
    img = tf.cast(img, tf.float32)

    shape = tf.shape(img)
    w, h = tf.cast(shape[0], tf.int64), tf.cast(shape[1], tf.int64)

    mask = smooth_mask(mask.cpu().numpy().astype(float))
    mask = tf.reduce_mean(mask, -1)

    img = img * tf.cast(mask > 0.01, tf.float32)[:, :, None]

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
    # img_max_zoom2 = img[tf.math.maximum(anchor_max_x-SIZE//2, 0): tf.math.minimum(anchor_max_x+SIZE//2, w),
    #                    tf.math.maximum(anchor_max_y-SIZE//2, 0): tf.math.minimum(anchor_max_y+SIZE//2, h)]
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


def segmentation_augmentation(batch_input, batch_size):
    seg_model = tf.keras.models.load_model(
        "segmentor_model/segmentation_model_576.h5",
        custom_objects={
            "binary_crossentropy_plus_jaccard_loss": sm.losses.binary_focal_jaccard_loss,
            "iou_score": sm.metrics.iou_score,
        },
    )
    seg_preprocess_input = sm.get_preprocessing("efficientnetb0")

    X = tf.image.resize(batch_input, (SIZE, SIZE))
    predicted_mask = one_step_inference(X)

    # X = seg_preprocess_input(batch_input).numpy()
    # out = seg_model.predict(X, batch_size=batch_size)
    mask = predicted_mask > 0.05
    samples = pi(img, mask)

    return samples


def segmentation_sam(batch_input, batch_labels, batch_domain, batch_size):
    X = tf.image.resize_with_pad(batch_input, SIZE, SIZE)
    samples = []
    labels = []
    domains = []
    for x, y, d in zip(X, batch_labels, batch_domain):
        predicted_mask = one_step_inference(x)
        # X = seg_preprocess_input(batch_input).numpy()
        # out = seg_model.predict(X, batch_size=batch_size)

        mask = predicted_mask > 0.2
        mask = mask[0]
        sample = pi(x, mask)
        label = tf.repeat([y], 4, axis=0)
        samples.append(sample)
        labels.append(label)
        domains.append(tf.repeat([d], 4, axis=0))

    return (
        tf.stack(samples, axis=0),
        tf.stack(labels, axis=0),
        tf.stack(domains, axis=0),
    )


def store_metric(store, key, value):
    if len(value.shape) == 0:
        value = tf.expand_dims(value, 0)
    store[key] = value if store[key] is None else tf.concat([store[key], value], axis=0)


def store_object(store, key, value):
    if len(value.shape) == 0:
        value = tf.expand_dims(value, 0)
    store[key] = value if store[key] is None else tf.concat([store[key], value], axis=0)


def print_metric(store):
    s = ""
    for key in store.keys():
        s += f" || {key}: {tf.reduce_mean(store[key])}"
    print(s)


def topK_metrict(values):
    top3 = []
    top5 = []
    for i, logit in enumerate(values["logits"]):
        sort_logits = np.argsort(logit)
        if values["labels"][i] in sort_logits[-3:]:
            top3.append(values["labels"][i])
        else:
            top3.append(sort_logits[-1])
        if values["labels"][i] in sort_logits[-5:]:
            top5.append(values["labels"][i])
        else:
            top5.append(sort_logits[-1])
    return top3, top5


def print_report(values, directory, epoch_i):
    top3, top5 = topK_metrict(values)

    print("TOP 1 : /n")
    cf1 = classification_report(
        values["labels"], values["predictions"], output_dict=True
    )
    cf1 = pd.DataFrame.from_dict(cf1)
    cf1.to_csv(os.path.join(directory, "Classification_Report_top1%04d.csv" % epoch_i))
    print(classification_report(values["labels"], values["predictions"]))
    print("TOP 3: /n")
    cf3 = classification_report(values["labels"], top3, output_dict=True)
    cf3 = pd.DataFrame.from_dict(cf3)
    cf3.to_csv(os.path.join(directory, "Classification_Report_top3%04d.csv" % epoch_i))
    print(classification_report(values["labels"], top3))
    print("TOP 5: /n")
    cf5 = classification_report(values["labels"], top5, output_dict=True)
    cf5 = pd.DataFrame.from_dict(cf5)
    cf5.to_csv(os.path.join(directory, "Classification_Report_top5%04d.csv" % epoch_i))
    print(classification_report(values["labels"], top5))


@tf.function
def _step_train(
    triplet_model, batch_inputs, batch_labels, batch_domain, domain_loss="in_out"
):
    with tf.GradientTape() as tape:
        batch_embedding, logits = triplet_model(batch_inputs, training=True)

        triplet_loss_class = batch_hard_triplet_loss(
            tf.argmax(batch_labels, -1), batch_embedding, MARGIN
        )
        if domain_loss == "in_out":
            triplet_loss_in_domain = batch_hard_in_domain_triplet_loss(
                tf.argmax(batch_labels, -1), batch_embedding, MARGIN, batch_domain
            )
            triplet_loss_out_domain = batch_hard_in_domain_triplet_loss(
                tf.argmax(batch_labels, -1), batch_embedding, MARGIN, batch_domain
            )
            triplet_loss_xdomain = (
                triplet_loss_in_domain * 0.5 + triplet_loss_out_domain * 0.5
            )
        elif domain_loss == "no":
            triplet_loss_in_domain = batch_hard_in_domain_triplet_loss(
                tf.argmax(batch_labels, -1), batch_embedding, MARGIN, batch_domain
            )
            triplet_loss_out_domain = batch_hard_in_domain_triplet_loss(
                tf.argmax(batch_labels, -1), batch_embedding, MARGIN, batch_domain
            )
            triplet_loss_xdomain = (
                triplet_loss_in_domain * 0.0 + triplet_loss_out_domain * 0.0
            )
        else:
            triplet_loss_xdomain = batch_hard_domain_triplet_loss(
                tf.argmax(batch_labels, -1), batch_embedding, MARGIN, batch_domain
            )
        cce_loss = CrossEntropy(batch_labels, logits)
        loss = (
            cce_loss
            + triplet_loss_class * LAMBDA_TRIPLET_CLASS
            + triplet_loss_xdomain * LAMBDA_TRIPLET_XDOMAIN
        )

    grads = tape.gradient(loss, triplet_model.trainable_weights)

    return grads, triplet_loss_class, triplet_loss_xdomain, cce_loss, logits


def train(
    triplet_model,
    optimizer,
    train_ds,
    test_ds,
    manager,
    directory,
    domain_loss="in_out",
    semantic_aug=False,
    epochs=50,
    starting_from=0,
    batch_size=10,
    prob_augmentation=0.9,
    val_ds=None,
):
    # optimizer =  tf.compat.v1.train.MomentumOptimizer(LR, momentum=0.9)#tf.keras.optimizers.Adam(LR)

    curves = {"train": [], "test": []}
    for epoch_i in range(starting_from, epochs):
        train_metrics = {
            "train_acc": None,
            "train_cce": None,
            "train_triplet_class": None,
            "train_triplet_xdomain": None,
        }

        train_acc = None
        c = 0

        for batch_inputs, batch_labels, batch_domain in train_ds:
            if np.random.uniform(0, 1) > prob_augmentation and semantic_aug:
                batch_inputs = batch_inputs.numpy()
                domain_indx = batch_domain == 0
                if domain_indx.numpy().sum() > 0:
                    batch_inputs_seg = segmentation_augmentation(
                        batch_inputs[batch_domain == 0, :, :, :] * 255, batch_size
                    )

                    batch_inputs[batch_domain == 0, :, :, :] = batch_inputs_seg / 255.0

                batch_inputs = tf.convert_to_tensor(batch_inputs)
            # optim loop
            (
                grads,
                triplet_loss_class,
                triplet_loss_xdomain,
                cce_loss,
                logits,
            ) = _step_train(
                triplet_model,
                batch_inputs,
                batch_labels,
                batch_domain,
                domain_loss=domain_loss,
            )
            optimizer.apply_gradients(zip(grads, triplet_model.trainable_weights))
            # store metrics

            # tf.print(f"Printing Labels to check {tf.argmax(batch_labels,-1)}")
            batch_acc = tf.cast(
                tf.argmax(logits, -1) == tf.argmax(batch_labels, -1), tf.float32
            )
            # tf.print(f" inputs {tf.math.reduce_min(batch_inputs[:5])} {tf.math.reduce_max(batch_inputs[:5])}")
            # tf.print(f" Logits {logits[:5]}")
            # tf.print(f" Predictions {tf.argmax(logits, -1)}")
            # tf.print(f" Labels {tf.argmax(batch_labels, -1)} ")
            store_metric(train_metrics, "train_acc", batch_acc)
            store_metric(train_metrics, "train_cce", cce_loss)
            store_metric(train_metrics, "train_triplet_class", triplet_loss_class)
            store_metric(train_metrics, "train_triplet_xdomain", triplet_loss_xdomain)
            curves["train"].append(
                [
                    epoch_i,
                    c,
                    tf.reduce_mean(batch_acc),
                    tf.reduce_mean(cce_loss),
                    tf.reduce_mean(triplet_loss_class),
                    tf.reduce_mean(triplet_loss_xdomain),
                ]
            )

            # regularly print metrics
            c += 1
            if c % 200 == 0:
                print(directory)
                print(f"Epoch {epoch_i} iteration {c}")
                # print(triplet_loss_class,triplet_loss_xdomain,cce_loss)
                print_metric(train_metrics)
                triplet_model.save(os.path.join(directory, f"model-{epoch_i}.h5"))
                manager.save(checkpoint_number=epoch_i)

        test_metrics = {
            "test_acc": None,
            "test_acc_top3": None,
            "test_acc_top5": None,
        }
        values = {
            "predictions": None,
            "labels": None,
            "predictions_top3": None,
            "predictions_top5": None,
            "logits": None,
        }
        print("testing starting ")
        for batch_inputs, batch_labels, batch_domain in test_ds:
            _, logits = triplet_model(batch_inputs)
            pred_batch = tf.argmax(logits, -1)
            batch_labels_ohe = tf.argmax(batch_labels, -1)
            batch_acc = tf.cast(pred_batch == batch_labels_ohe, tf.float32)
            batch_acc_3 = tf.cast(
                tf.math.in_top_k(batch_labels_ohe, logits, 3), tf.float32
            )
            batch_acc_5 = tf.cast(
                tf.math.in_top_k(batch_labels_ohe, logits, 5), tf.float32
            )

            store_object(values, "predictions", pred_batch.numpy())
            store_object(values, "labels", batch_labels_ohe.numpy())
            store_object(values, "logits", logits.numpy())

            store_metric(test_metrics, "test_acc", batch_acc)
            store_metric(test_metrics, "test_acc_top3", batch_acc_3)
            store_metric(test_metrics, "test_acc_top5", batch_acc_5)

            manager.save(checkpoint_number=epoch_i)
        preds = values["predictions"].numpy()
        gts = values["labels"].numpy()
        np.save(os.path.join(directory, f"{epoch_i}_preds.npy"), preds)
        np.save(os.path.join(directory, f"{epoch_i}_labels.npy"), gts)

        print("------Directory-----")
        print("/n")
        print(directory)
        print("Accuracy:")
        print("/n")
        curves["test"].append(
            [
                epoch_i,
                tf.reduce_mean(test_metrics["test_acc"]),
                tf.reduce_mean(test_metrics["test_acc_top3"]),
                tf.reduce_mean(test_metrics["test_acc_top5"]),
            ]
        )

        print_metric(test_metrics)
        print_report(values, directory, epoch_i)

        # with open(os.path.join(directory,f'{epoch_i}_conf_matrix.json'), "w") as outfile:
        #    json.dump(values, outfile)
    traindf_metrics = pd.DataFrame(
        curves["train"],
        columns=[
            "epoch",
            "iteration",
            "batch_acc",
            "cce_loss",
            "triplet_loss_class",
            "triplet_loss_xdomain",
        ],
    )
    traindf_metrics.to_csv(os.path.join(directory, "train_logger.csv"))
    testdf_metrics = pd.DataFrame(
        curves["test"], columns=["epoch", "test_acc", "test_acc_top3", "test_acc_top5"]
    )
    testdf_metrics.to_csv(os.path.join(directory, "test_logger.csv"))


def train_sam(
    triplet_model,
    optimizer,
    train_ds,
    test_ds,
    manager,
    directory,
    n_classes=55,
    domain_loss="in_out",
    semantic_aug=False,
    epochs=50,
    starting_from=0,
    batch_size=10,
    prob_augmentation=0.9,
    val_ds=None,
):
    # optimizer =  tf.compat.v1.train.MomentumOptimizer(LR, momentum=0.9)#tf.keras.optimizers.Adam(LR)

    curves = {"train": [], "test": []}
    for epoch_i in range(starting_from, epochs):
        train_metrics = {
            "train_acc": None,
            "train_cce": None,
            "train_triplet_class": None,
            "train_triplet_xdomain": None,
        }

        train_acc = None
        c = 0

        for batch_inputs, batch_labels, batch_domain in train_ds:
            if np.random.uniform(0, 1) > prob_augmentation and semantic_aug:
                batch_inputs = batch_inputs.numpy()
                domain_indx = batch_domain == 0

                batch_inputs_seg, batch_labels, domains = segmentation_sam(
                    batch_inputs * 255, batch_labels, batch_domain, batch_size
                )

                batch_inputs = batch_inputs_seg / 255.0
                batch_inputs = tf.reshape(batch_inputs, (-1, SIZE, SIZE, 3))

                batch_inputs = tf.convert_to_tensor(batch_inputs)
                batch_labels = tf.reshape(batch_labels, (-1, n_classes))

                batch_domain = tf.reshape(domains, -1)

            # optim loop
            (
                grads,
                triplet_loss_class,
                triplet_loss_xdomain,
                cce_loss,
                logits,
            ) = _step_train(
                triplet_model,
                batch_inputs,
                batch_labels,
                batch_domain,
                domain_loss=domain_loss,
            )
            optimizer.apply_gradients(zip(grads, triplet_model.trainable_weights))
            # store metrics
            labels = tf.math.reduce_sum(batch_labels, axis=0)
            labels = tf.argmax(labels, -1)

            logits = tf.math.reduce_sum(logits, axis=0)
            logits = tf.argmax(logits, -1)
            # tf.print(f"Printing Labels to check {tf.argmax(batch_labels,-1)}")
            batch_acc = tf.cast(labels == logits, tf.float32)
            # tf.print(f" inputs {tf.math.reduce_min(batch_inputs[:5])} {tf.math.reduce_max(batch_inputs[:5])}")
            # tf.print(f" Logits {logits[:5]}")
            # tf.print(f" Predictions {tf.argmax(logits, -1)}")
            # tf.print(f" Labels {tf.argmax(batch_labels, -1)} ")
            store_metric(train_metrics, "train_acc", batch_acc)
            store_metric(train_metrics, "train_cce", cce_loss)
            store_metric(train_metrics, "train_triplet_class", triplet_loss_class)
            store_metric(train_metrics, "train_triplet_xdomain", triplet_loss_xdomain)
            curves["train"].append(
                [
                    epoch_i,
                    c,
                    tf.reduce_mean(batch_acc),
                    tf.reduce_mean(cce_loss),
                    tf.reduce_mean(triplet_loss_class),
                    tf.reduce_mean(triplet_loss_xdomain),
                ]
            )

            # regularly print metrics
            c += 1
            if c % 200 == 0:
                print(directory)
                print(f"Epoch {epoch_i} iteration {c}")
                # print(triplet_loss_class,triplet_loss_xdomain,cce_loss)
                print_metric(train_metrics)
                triplet_model.save(os.path.join(directory, f"model-{epoch_i}.h5"))
                manager.save(checkpoint_number=epoch_i)

        test_metrics = {
            "test_acc": None,
            "test_acc_top3": None,
            "test_acc_top5": None,
        }
        values = {
            "predictions": None,
            "labels": None,
            "predictions_top3": None,
            "predictions_top5": None,
            "logits": None,
        }
        print("testing starting ")
        for batch_inputs, batch_labels, batch_domain in test_ds:
            _, logits = triplet_model(batch_inputs)
            pred_batch = tf.argmax(logits, -1)
            batch_labels_ohe = tf.argmax(batch_labels, -1)
            batch_acc = tf.cast(pred_batch == batch_labels_ohe, tf.float32)
            batch_acc_3 = tf.cast(
                tf.math.in_top_k(batch_labels_ohe, logits, 3), tf.float32
            )
            batch_acc_5 = tf.cast(
                tf.math.in_top_k(batch_labels_ohe, logits, 5), tf.float32
            )

            store_object(values, "predictions", pred_batch.numpy())
            store_object(values, "labels", batch_labels_ohe.numpy())
            store_object(values, "logits", logits.numpy())

            store_metric(test_metrics, "test_acc", batch_acc)
            store_metric(test_metrics, "test_acc_top3", batch_acc_3)
            store_metric(test_metrics, "test_acc_top5", batch_acc_5)

            manager.save(checkpoint_number=epoch_i)
        preds = values["predictions"].numpy()
        gts = values["labels"].numpy()
        np.save(os.path.join(directory, f"{epoch_i}_preds.npy"), preds)
        np.save(os.path.join(directory, f"{epoch_i}_labels.npy"), gts)

        print("------Directory-----")
        print("/n")
        print(directory)
        print("Accuracy:")
        print("/n")
        curves["test"].append(
            [
                epoch_i,
                tf.reduce_mean(test_metrics["test_acc"]),
                tf.reduce_mean(test_metrics["test_acc_top3"]),
                tf.reduce_mean(test_metrics["test_acc_top5"]),
            ]
        )

        print_metric(test_metrics)
        print_report(values, directory, epoch_i)

        # with open(os.path.join(directory,f'{epoch_i}_conf_matrix.json'), "w") as outfile:
        #    json.dump(values, outfile)
    traindf_metrics = pd.DataFrame(
        curves["train"],
        columns=[
            "epoch",
            "iteration",
            "batch_acc",
            "cce_loss",
            "triplet_loss_class",
            "triplet_loss_xdomain",
        ],
    )
    traindf_metrics.to_csv(os.path.join(directory, "train_logger.csv"))
    testdf_metrics = pd.DataFrame(
        curves["test"], columns=["epoch", "test_acc", "test_acc_top3", "test_acc_top5"]
    )
    testdf_metrics.to_csv(os.path.join(directory, "test_logger.csv"))


import gc


def leave_one_out_experiment(
    shots=[1, 5, 10],
    classes=19,
    domain_loss="xdomain",
    DATA=["gan_leaves_fossils_fewshot_v1.0"],
    pretrained_herbarium=False,
    prefix="22",
    epochs=10,
    gray=False,
    backbone=tf.keras.applications.ResNet50V2,
    batch_size=18,
    semantic_aug=True,
    contrastive=False,
):
    NUMBER_CLASSES = 55
    for d in DATA:
        for shot in shots:
            for label in range(classes):
                triplet_model = get_triplet_model(
                    input_shape=(SIZE, SIZE, 3),
                    embedding_units=256,
                    embedding_depth=2,
                    backbone_class=backbone,
                    nb_classes=NUMBER_CLASSES,
                    load_weights=pretrained_herbarium,
                )

                # ptimizer = tf.keras.optimizers.Adam(LR)
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=LR, momentum=0.9, name="SGD"
                )
                if pretrained_herbarium:
                    NAME = (
                        f"{prefix}_PRE_SGD_0.3_{domain_loss}_{d}_{shot}_Label_{label}"
                    )
                else:
                    NAME = f"{prefix}_SGD_0.3_{domain_loss}_{d}_{shot}_Label_{label}"
                checkpointdir = os.path.join(CKPT_DIRECTORY, NAME)
                os.makedirs(checkpointdir, exist_ok=True)
                checkpoint = tf.train.Checkpoint(
                    optimizer=optimizer, model=triplet_model
                )
                manager = tf.train.CheckpointManager(
                    checkpoint, directory=checkpointdir, max_to_keep=5
                )
                if manager.latest_checkpoint:
                    starting_from = int(
                        manager.latest_checkpoint.split("/")[-1].split("-")[-1]
                    )
                    print(f"INFO: restarting training from {starting_from}")
                    if starting_from >= epochs:
                        print("This model seem that was already trained")
                        print("Skipping")
                        continue
                else:
                    starting_from = 0

                status = checkpoint.restore(manager.latest_checkpoint)
                print(status)
                label_out = label
                shot = shot
                print
                train_ds, val_ds, test_ds = leaves_fewshot_ds(
                    label_out,
                    shot,
                    batch_size=batch_size,
                    dataname=d,
                    gray=gray,
                    wsize=SIZE,
                    hsize=SIZE,
                    number_classes=NUMBER_CLASSES,
                )
                # p = mp.Process(target=train,args=(triplet_model, optimizer, train_ds, test_ds,manager,checkpointdir))
                # p.start()
                # p.join()
                # print('subprocess is done')

                train(
                    triplet_model,
                    optimizer,
                    train_ds,
                    test_ds,
                    manager,
                    checkpointdir,
                    domain_loss=domain_loss,
                    epochs=epochs,
                    starting_from=starting_from,
                    batch_size=batch_size,
                    semantic_aug=semantic_aug,
                )
                tf.keras.backend.clear_session()
                gc.collect()
                del triplet_model


def unseen_experiment():
    NUMBER_CLASSES = 171
    pretrained_herbarium = True
    segmentation_augmentation = True
    domain_loss = "xdomain"
    finer_model = True
    triplet_model = get_triplet_model(
        input_shape=(SIZE, SIZE, 3),
        embedding_units=256,
        embedding_depth=2,
        backbone_class=tf.keras.applications.ResNet50V2,
        nb_classes=NUMBER_CLASSES,
        load_weights=pretrained_herbarium,
        finer_model=finer_model,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, name="SGD")

    NAME = f"Unseen_biggest_brunei_all_4_finer_{SIZE}_{domain_loss}"
    if pretrained_herbarium:
        NAME += "_herbarium_"
    if segmentation_augmentation:
        NAME += "_segmentation"
    checkpointdir = os.path.join(CKPT_DIRECTORY, NAME)
    os.makedirs(checkpointdir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=triplet_model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpointdir, max_to_keep=5
    )
    status = checkpoint.restore(manager.latest_checkpoint)
    print(status)

    batch_size = 20
    train_ds, val_ds, test_ds = leaves_fewshot_ds(
        label_out=None,
        shot=None,
        wsize=SIZE,
        hsize=SIZE,
        batch_size=batch_size,
        dataname="gan_all_leaves_fossils_v1.1_processed_brunei",
        number_classes=NUMBER_CLASSES,
    )
    # p = mp.Process(target=train,args=(triplet_model, optimizer, train_ds, test_ds,manager,checkpointdir))
    # p.start()
    # p.join()
    # print('subprocess is done')

    train(
        triplet_model,
        optimizer,
        train_ds,
        test_ds,
        manager,
        checkpointdir,
        domain_loss=domain_loss,
        batch_size=batch_size,
        semantic_aug=segmentation_augmentation,
        prob_augmentation=0.5,
    )

    pass


def leaves():
    SIZE = 512
    NUMBER_CLASSES = 19
    pretrained_herbarium = False
    segmentation_augmentation = False
    domain_loss = "xdomain"
    finer_model = True
    domain_loss = "no"
    database = "leaves_fossils_v1.1_processed"
    triplet_model = get_triplet_model(
        input_shape=(SIZE, SIZE, 3),
        embedding_units=256,
        embedding_depth=2,
        backbone_class=tf.keras.applications.ResNet50V2,
        nb_classes=NUMBER_CLASSES,
        load_weights=pretrained_herbarium,
        finer_model=finer_model,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, name="SGD")
    print("Model Defined")
    NAME = f"GAN_Leaves_Fossils_{domain_loss}_triplet_control"
    checkpointdir = os.path.join(CKPT_DIRECTORY, NAME)
    os.makedirs(checkpointdir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=triplet_model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpointdir, max_to_keep=5
    )
    status = checkpoint.restore(manager.latest_checkpoint)
    print(status)

    batch_size = 8
    print("Launching training ... ")
    train_ds, val_ds, test_ds = leaves_fewshot_ds(
        label_out=None,
        shot=None,
        wsize=SIZE,
        hsize=SIZE,
        batch_size=batch_size,
        dataname=database,
        number_classes=NUMBER_CLASSES,
    )
    # p = mp.Process(target=train,args=(triplet_model, optimizer, train_ds, test_ds,manager,checkpointdir))
    # p.start()
    # p.join()
    # print('subprocess is done')

    train(
        triplet_model,
        optimizer,
        train_ds,
        test_ds,
        manager,
        checkpointdir,
        domain_loss=domain_loss,
    )


def leaves_50():
    SIZE = 576
    NUMBER_CLASSES = 55
    epochs = 50
    pretrained_herbarium = False
    segmentation_augmentation = True
    finer_model = True
    domain_loss = "xdomain"
    database = "leaves_paper_2022_50_50"
    NAME = f"Leaves_BEIT_{domain_loss}_triplet_control_{database}_finer"

    if pretrained_herbarium:
        NAME += "_herbarium_"
    if segmentation_augmentation:
        NAME += "_segmentation"
    triplet_model = get_triplet_model_beit(
        input_shape=(SIZE, SIZE, 3),
        embedding_units=256,
        embedding_depth=2,
        nb_classes=NUMBER_CLASSES,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, name="SGD")
    print("Model Defined")

    checkpointdir = os.path.join(CKPT_DIRECTORY, NAME)
    os.makedirs(checkpointdir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=triplet_model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpointdir, max_to_keep=5
    )
    status = checkpoint.restore(manager.latest_checkpoint)
    print(status)

    batch_size = 8
    print("Launching training ... ")
    train_ds, val_ds, test_ds = leaves_fewshot_ds(
        label_out=None,
        shot=None,
        wsize=SIZE,
        hsize=SIZE,
        batch_size=batch_size,
        dataname=database,
        number_classes=NUMBER_CLASSES,
    )
    # p = mp.Process(target=train,args=(triplet_model, optimizer, train_ds, test_ds,manager,checkpointdir))
    # p.start()
    # p.join()
    # print('subprocess is done')

    train(
        triplet_model,
        optimizer,
        train_ds,
        test_ds,
        manager,
        checkpointdir,
        epochs=epochs,
        domain_loss=domain_loss,
        semantic_aug=segmentation_augmentation,
        prob_augmentation=0.8,
    )


def fossils_19_sam(
    database="leaves_paper_2022_50_50", semantic_aug=False, contrastive=False, epochs=50
):
    SIZE = 384
    NUMBER_CLASSES = 19
    epochs = epochs
    pretrained_herbarium = False
    segmentation_augmentation = semantic_aug
    finer_model = False
    domain_loss = "xdomain"

    NAME = f"Leaves_BEIT_{domain_loss}_triplet_control_{database}_{SIZE}"

    if pretrained_herbarium:
        NAME += "_herbarium_"
    if segmentation_augmentation:
        NAME += "_segmentation"
    triplet_model = get_triplet_model_beit(
        input_shape=(SIZE, SIZE, 3),
        embedding_units=256,
        embedding_depth=2,
        nb_classes=NUMBER_CLASSES,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, name="SGD")
    print("Model Defined")

    checkpointdir = os.path.join(CKPT_DIRECTORY, NAME)
    os.makedirs(checkpointdir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=triplet_model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpointdir, max_to_keep=5
    )
    status = checkpoint.restore(manager.latest_checkpoint)
    print(status)

    batch_size = 4
    print("Launching training ... ")
    train_ds, val_ds, test_ds = leaves_fewshot_ds(
        label_out=None,
        shot=None,
        wsize=SIZE,
        hsize=SIZE,
        batch_size=batch_size,
        dataname=database,
        number_classes=NUMBER_CLASSES,
    )
    # p = mp.Process(target=train,args=(triplet_model, optimizer, train_ds, test_ds,manager,checkpointdir))
    # p.start()
    # p.join()
    # print('subprocess is done')

    train_sam(
        triplet_model,
        optimizer,
        train_ds,
        test_ds,
        manager,
        checkpointdir,
        epochs=epochs,
        n_classes=NUMBER_CLASSES,
        domain_loss=domain_loss,
        semantic_aug=segmentation_augmentation,
        prob_augmentation=0.8,
    )


def leaves_50_mum(semantic_aug=False, contrastive=False, epochs=50):
    SIZE = 384
    NUMBER_CLASSES = 170
    epochs = epochs
    pretrained_herbarium = False
    segmentation_augmentation = semantic_aug
    finer_model = False
    domain_loss = "xdomain"
    database = "mummified"
    NAME = f"Leaves_BEIT_{domain_loss}_triplet_control_{database}"

    if pretrained_herbarium:
        NAME += "_herbarium_"
    if segmentation_augmentation:
        NAME += "_segmentation"
    triplet_model = get_triplet_model_beit(
        input_shape=(SIZE, SIZE, 3),
        embedding_units=256,
        embedding_depth=2,
        nb_classes=NUMBER_CLASSES,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, name="SGD")
    print("Model Defined")

    checkpointdir = os.path.join(CKPT_DIRECTORY, NAME)
    os.makedirs(checkpointdir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=triplet_model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpointdir, max_to_keep=5
    )
    status = checkpoint.restore(manager.latest_checkpoint)
    print(status)

    batch_size = 4
    print("Launching training ... ")
    train_ds, val_ds, test_ds = leaves_fewshot_ds(
        label_out=None,
        shot=None,
        wsize=SIZE,
        hsize=SIZE,
        batch_size=batch_size,
        dataname=database,
        number_classes=NUMBER_CLASSES,
    )
    # p = mp.Process(target=train,args=(triplet_model, optimizer, train_ds, test_ds,manager,checkpointdir))
    # p.start()
    # p.join()
    # print('subprocess is done')

    train_sam(
        triplet_model,
        optimizer,
        train_ds,
        test_ds,
        manager,
        checkpointdir,
        epochs=epochs,
        domain_loss=domain_loss,
        n_classes=NUMBER_CLASSES,
        semantic_aug=segmentation_augmentation,
        prob_augmentation=0.8,
    )


def mummified():
    SIZE = 576
    NUMBER_CLASSES = 170
    epochs = 100
    pretrained_herbarium = False
    segmentation_augmentation = False
    finer_model = True
    domain_loss = "xdomain"
    database = "mummified"
    NAME = f"Mummified_{domain_loss}_triplet_control_{database}_finer"

    if pretrained_herbarium:
        NAME += "_herbarium_"
    if segmentation_augmentation:
        NAME += "_segmentation"
    triplet_model = get_triplet_model(
        input_shape=(SIZE, SIZE, 3),
        embedding_units=256,
        embedding_depth=2,
        backbone_class=tf.keras.applications.ResNet50V2,
        nb_classes=NUMBER_CLASSES,
        load_weights=pretrained_herbarium,
        finer_model=finer_model,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, name="SGD")
    print("Model Defined")

    checkpointdir = os.path.join(CKPT_DIRECTORY, NAME)
    os.makedirs(checkpointdir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=triplet_model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpointdir, max_to_keep=5
    )
    status = checkpoint.restore(manager.latest_checkpoint)
    print(status)

    batch_size = 8
    print("Launching training ... ")
    train_ds, val_ds, test_ds = leaves_fewshot_ds(
        label_out=None,
        shot=None,
        wsize=SIZE,
        hsize=SIZE,
        batch_size=batch_size,
        dataname=database,
        number_classes=NUMBER_CLASSES,
    )
    # p = mp.Process(target=train,args=(triplet_model, optimizer, train_ds, test_ds,manager,checkpointdir))
    # p.start()
    # p.join()
    # print('subprocess is done')

    train(
        triplet_model,
        optimizer,
        train_ds,
        test_ds,
        manager,
        checkpointdir,
        epochs=epochs,
        domain_loss=domain_loss,
        semantic_aug=segmentation_augmentation,
        prob_augmentation=0.8,
    )


# mummified()
# leaves_50()
# leaves()
# unseen_experiment()
# leaves()
# print('starting leave one out')
shots = [0]
classes = 19
domain_loss = "xdomain"
# DATA=['gan_leaves_fossils_v1.1_processed_2022_gan_leave_one_out3']

DATA = ["2023_florissant_gan_leaves_fossils"]
# DATA=['mummified']
pretrained_herbarium = False
semantic_aug = True
contrastive = True
if semantic_aug:
    prefix = "19_2023_seg_"
else:
    prefix = "170_2023_contrastive"
epochs = 100
# leave_one_out_experiment(shots,classes,domain_loss,DATA,pretrained_herbarium,prefix,semantic_aug=semantic_aug,contrastive=contrastive)
# unseen_experiment()
##leaves()
for d in DATA:
    fossils_19_sam(database=d, semantic_aug=semantic_aug, epochs=100)
