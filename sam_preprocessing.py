import torch

# torch.cuda.set_per_process_memory_fraction(0.2, device=0)

import numpy as np
import torch.nn.functional as F
import tensorflow as tf

from leaves_processing import pi
from segment_anything import sam_model_registry, SamPredictor

import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm

import cv2

import ipdb


def get_segmentation_model():
    sam = sam_model_registry["default"]("./models/sam_02-06_dice_mse_0.pth")
    sam = sam.cuda()
    predictor = SamPredictor(sam)
    ##put model on cuda
    return sam, predictor


def save_masked_image(images, class_name):
    print(len(images))
    print(images[0].shape)
    masked_images = cv2.hconcat(images)
    cv2.imwrite(
        f"./masked_images/{class_name}.png",
        cv2.cvtColor(masked_images, cv2.COLOR_RGB2BGR),
    )
    print("saved masked images")


def preprocess(img, sam, predictor):
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


def one_step_inference(x, sam, predictor):
    if len(x.shape) == 3:
        original_size = x.shape[:2]
    elif len(x.shape) == 4:
        original_size = x.shape[1:3]

    x, intermediate_shape = preprocess(x, sam, predictor)

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


SIZE = 384


def segmentation_sam(batch_input, batch_labels, batch_domain, sam, predictor):
    X = tf.image.resize_with_pad(batch_input, SIZE, SIZE)
    samples = []
    labels = []
    domains = []
    images = []
    zoom_images = []
    save_samples = []
    print(X.shape, len(batch_labels))
    for x, y, d in zip(X, batch_labels, batch_domain):
        x = x * 255
        predicted_mask = one_step_inference(x, sam, predictor)
        # # #X = seg_preprocess_input(batch_input).numpy()
        # # #out = seg_model.predict(X, batch_size=batch_size)

        mask = predicted_mask > 0.1
        mask = mask[0]
        sample = pi(x, mask)
        # label = tf.repeat([y],4,axis=0)
        images.append(sample[0])
        zoom_images.append(sample[0])
        labels.append(tf.one_hot(int(y), 142))
        domains.append(tf.repeat([d], 4, axis=0))

    for i in range(4):
        save_samples.append(images[i].numpy())
    print(save_samples[0].shape)
    save_masked_image(save_samples, batch_labels[0])

    return (
        tf.stack(images, axis=0),
        tf.stack(zoom_images, axis=0),
        labels,
        tf.stack(domains, axis=0),
    )


# def segmentation_sam(batch_input,batch_labels,batch_domain, sam, predictor):

#     batch_size = batch_input.shape[0]
#     print(type(batch_input[0]))

#     # print(X[0])
#     # X = torch.tensor(batch_input)
#     # print(X.shape)
#     samples =[]
#     save_samples = []
#     labels =[]
#     domains =[]
#     zoom_images = []
#     for i in range(batch_size):
#         x,y,d = batch_input[i]*255, batch_labels[i], batch_domain[i]
#         ipdb.set_trace()
#         predicted_mask = one_step_inference(x, sam, predictor)
#         #X = seg_preprocess_input(batch_input).numpy()
#         #out = seg_model.predict(X, batch_size=batch_size)
#         # print(predicted_mask.shape)
#         mask = predicted_mask>0.1
#         mask = mask[0]
#         # mask = mask.astype(np.float32)[0][0]
#         # mask = np.expand_dims(mask, axis = -1)
#         # x = x.cpu().numpy().astype(np.uint8)
#         # print('segmentation_sam')
#         # print(X[i])
#         # print('Before PI')
#         # print(X[i].numpy().max(), X[i].numpy().min())
#         sample = pi(x, mask)
#         # print('After PI')
#         # print(sample[0].numpy().max(), sample[0].numpy().min())
#         samples.append(sample[0])
#         labels.append(tf.one_hot(int(y), 142))
#         domains.append(tf.repeat([d],4,axis=0))
#         zoom_images.append(sample[3])
#         # print(sample.shape)

#     for i in range(4):
#        save_samples.append(samples[i].numpy())
#     print(save_samples[0].shape)
#     save_masked_image(save_samples, labels[0])
#     # ipdb.set_trace()
#     return tf.stack(samples,axis=0), tf.stack(zoom_images, axis = 0), labels, tf.stack(domains,axis=0)


def segmentation_augmentation(batch_input, batch_labels, batch_size=None):
    # seg_model = tf.keras.models.load_model('segmentor_model/segmentation_model_576.h5',
    #                                custom_objects={'binary_crossentropy_plus_jaccard_loss': sm.losses.binary_focal_jaccard_loss,
    #                                                'iou_score': sm.metrics.iou_score})

    # seg_preprocess_input = sm.get_preprocessing("efficientnetb0")
    # x = tf.image.resize(batch_input, (SIZE, SIZE))
    batch_size = batch_input.shape[0]
    X = torch.tensor(batch_input)
    all_samples = []
    labels = []
    for i in range(batch_size):
        x, y = X[i], batch_labels[i]
        predicted_mask = one_step_inference(x)
        # X = seg_preprocess_input(batch_input).numpy()
        # out = seg_model.predict(X, batch_size=batch_size)
        mask = predicted_mask > 0.05
        samples = pi(x, mask)
        all_samples.append(samples)
        labels.append(tf.one_hot(int(y), 142))

    return all_samples, labels
