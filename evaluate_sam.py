import torch
from sklearn.metrics import classification_report
from Craft.craft.new_craft_tf import Craft
import cv2

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
from data import mocking_ds, leaves_fewshot_ds
from sklearn.metrics import classification_report
from data import leaves_fewshot_ds
import pandas as pd

# import multiprocessing as mp
# import segmentation_models as sm
import json

# sm.set_framework('tf.keras')
# sm.framework()

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


def pi(img, mask):
    img = tf.cast(img, tf.float32)

    shape = tf.shape(img)
    w, h = tf.cast(shape[0], tf.int64), tf.cast(shape[1], tf.int64)

    # mask = mask.cpu().numpy().astype(float)[0]
    mask = mask.astype(float)[0]
    mask = smooth_mask_v2(mask, kernel_size=15)

    # mask = tf.reduce_mean(mask, -1)

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
    img_max_zoom2 = img[
        tf.math.maximum(anchor_max_x - SIZE // 2, 0) : tf.math.minimum(
            anchor_max_x + SIZE // 2, w
        ),
        tf.math.maximum(anchor_max_y - SIZE // 2, 0) : tf.math.minimum(
            anchor_max_y + SIZE // 2, h
        ),
    ]

    # cv2.imwrite('img.jpg',img_resize.numpy())
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


def segmentation_sam(batch_input, batch_labels, batch_domain, batch_number):
    X = tf.image.resize_with_pad(batch_input, SIZE, SIZE)
    samples = []
    labels = []
    domains = []
    start = batch_number * len(batch_input)
    for x, y, d in zip(X, batch_labels, batch_domain):
        predicted_mask = one_step_inference(x)
        # X = seg_preprocess_input(batch_input).numpy()
        # out = seg_model.predict(X, batch_size=batch_size)
        mask = predicted_mask > 0.1
        mask = mask[0]
        total_mask = mask.shape[0] * mask.shape[1]
        mask_sum = mask.sum()
        mask = mask.cpu().numpy()

        if d == 1:
            if mask_sum < total_mask * 0.4:
                mask = tf.ones_like(mask).numpy()
        else:
            if mask_sum < total_mask * 0.1:
                mask = tf.ones_like(mask).numpy()  # change later
        sample = pi(x, mask)
        samples.append(sample)
        labels.append(tf.math.argmax(y))
        domains.append(tf.repeat([d], 4, axis=0))
      
        x = np.hstack(sample)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'./sam_masks/image_{start}.png', x*255)
        start+=1
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


## iterate through data loader
def evaluate(model, test_ds, batch_size=10, prob_augmentation=0.9, val_ds=None):
    all_leaves, all_leaves_predictions = [], []
    all_fossils, all_fossils_predictions = [], []

    patch_size = 192
    craft = Craft(
        input_to_latent=g,
        latent_to_logit=h,
        number_of_concepts=20,
        patch_size=patch_size,
        batch_size=64,
    )

    for b, (batch_inputs, batch_labels, batch_domain) in enumerate(test_ds):
        images, labels, domains = segmentation_sam(
            batch_inputs, batch_labels, batch_domain, b
        )

        # input_images = images[:, 0, :, :]

        # labels = labels.numpy()
        # _, batch_logits = model.predict(input_images)

        # predictions = tf.math.top_k(batch_logits, k=5)

        # final_predictions = []

        # for i in range(len(predictions[1])):
        #     if labels[i] in predictions[1][i]:
        #         final_predictions.append(labels[i])
        #     else:
        #         final_predictions.append(predictions[1][i][0])
        # final_predictions = np.array(final_predictions)

        # all_leaves.extend(labels[batch_domain == 0])
        # all_fossils.extend(labels[batch_domain == 1])

        # all_leaves_predictions.extend(final_predictions[batch_domain == 0])
        # all_fossils_predictions.extend(final_predictions[batch_domain == 1])
        # print(f"Batch {b} done")

        # final_images = input_images[labels == final_predictions]
        # final_labels = labels[labels == final_predictions]

        # activations, patches = craft.fit(final_images)

        # grayscale_images = tf.reduce_mean(patches, axis=3, keepdims=True)
        # binary_images = tf.cast(tf.equal(grayscale_images, 0), tf.float32)
        # num_black_pixels = tf.reduce_sum(binary_images, axis=(1, 2, 3))
        # total_pixels = tf.cast(
        #     tf.reduce_prod(tf.shape(grayscale_images)[1:]), tf.float32
        # )
        # percentage_black_pixels = (num_black_pixels / total_pixels) * 100
        # patches = tf.gather(patches, tf.where(percentage_black_pixels < 95))[
        #     :, 0, :, :
        # ]
        # activations = tf.gather(activations, tf.where(percentage_black_pixels < 95))[
        #     :, 0, :, :
        # ]
        # for i in range(patches.shape[0]):
        #     os.makedirs("./patches_v2", exist_ok=True)
        #     import cv2

        #     cv2.imwrite(f"./patches_v3/patch_{i}.jpg", patches[i].numpy() * 255)
        # for i in range(final_images.shape[0]):
        #     cv2.imwrite(f"./patches_v3/image_{i}.jpg", final_images[i].numpy() * 255)

        # if b % 2 == 0:
        #     if b > 0:
        #         activations_and_patches = np.load(
        #             "./activations/activations_patches.npz"
        #         )
        #         prev_activations = activations_and_patches["activations"]
        #         prev_patches = activations_and_patches["patches"]
        #         activations = np.concatenate([prev_activations, activations])
        #         patches = np.concatenate([prev_patches, patches])
        #     np.savez(
        #         "./activations/activations_patches.npz",
        #         **{"activations": activations, "patches": patches},
        #     )
        #     print(f"Activations for saved till batch {b}")
        #     print(classification_report(all_leaves, all_leaves_predictions))
        #     print(classification_report(all_fossils, all_fossils_predictions))

        #     crops, crops_u, w = craft.activation_transform(activations, patches)

        #     print(
        #         f"crops shape: {crops.shape}, crops_u shape: {crops_u.shape}, w shape: {w.shape}"
        #     )
        #     importances = craft.estimate_importance(
        #         final_images, class_labels=final_labels
        #     )

        #     # images_u = craft.transform(images_fossils_correct)

        #     most_important_concepts = helpers.plot_new_histogram(
        #         importances, histogram_dir
        #     )
        #     helpers.save_new_crops(
        #         most_important_concepts,
        #         importances,
        #         crops_u,
        #         crops,
        #         save_crops,
        #         b
        #     )


if __name__ == "__main__":
    model_path = "./models/model-13.h5"
    csv_path = "./csv/fossils.csv"
    fossils_data_dir = (
        "/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Florissant_Fossil_v2.0"
    )
    leaves_data_dir = (
        "/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Extant_Leaves"
    )
    save_crops = "./crops/fossils_leaves_crops/exp5_RELU_192_20_v2"
    histogram_dir = "./histogram/exp5_RELU_192_20_v2"

    model, g, h = helpers.get_model(model_path)
    batch_size = 16
    SIZE = 384
    print("loading dataset")
    test_ds = leaves_fewshot_ds(
        label_out=None,
        shot=None,
        wsize=SIZE,
        hsize=SIZE,
        batch_size=batch_size,
        dataname="leaves_paper_2022_50_50",
        number_classes=142,
    )
    print("dataset loaded")
    evaluate(model, test_ds, batch_size=batch_size)
