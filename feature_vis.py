import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import xplique
from xplique.features_visualizations import Objective, optimize

from keras_cv_attention_models import beit

# mobilenet = tf.keras.applications.MobileNet()
beit = beit.BeitBasePatch16(input_shape=(224, 224, 3), pretrained="imagenet21k-ft1k")
# import ipdb;ipdb.set_trace()

classes = [
    (1, "Goldfish"),
    (33, "Loggerhead turtle"),
    (75, "Black widow"),
    (294, "Brown Bear"),
]

beit.layers[-1].activation = tf.keras.activations.linear

obj_logits = Objective.neuron(beit, -1, [c_id for c_id, c_name in classes])
imgs, _ = optimize(
    obj_logits,
    nb_steps=1024,  # number of iterations
    optimizer=tf.keras.optimizers.Adam(0.05),
)


plt.rcParams["figure.figsize"] = [12, 8]
for i in range(len(classes)):
    plt.subplot(len(classes) // 4, 4, i + 1)
    plt.imshow(imgs[0][i])
    plt.title(classes[i][1])
    plt.axis("off")
    plt.savefig("feature_vis_beit_imagenet.png")
