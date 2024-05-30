import numpy as np
import cv2

import tensorflow as tf
import numpy as np

import os
import numpy as np
from Craft.craft.new_craft_tf import Craft
import helpers

# from craft.craft_torch import Craft, torch_to_numpy

def main():
    
    root_dir = '/cifs/data/tserre_lrs/projects/prj_video_imagenet/mae/data/imagenet/val/n02325366'
    save_crops = "./crops/fossils_leaves_crops/imagenet_resenet50_80_10"
    histogram_dir = "./histogram/imagenet_resenet50_80_10"

    imgs_paths = os.listdir(root_dir)

    images = []
    labels = []
    for img in imgs_paths:
        im_path = os.path.join(root_dir, img)
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224))
        # image = tf.keras.applications.densenet.preprocess_input(image)
        images.append(image)
        labels.append(330)
    
    images = np.array(images)
    images_preprocessed = tf.keras.applications.densenet.preprocess_input(images)

    model = tf.keras.applications.DenseNet121(classifier_activation="linear")
    g = tf.keras.Model(model.input, model.layers[-3].output)
    h = tf.keras.Model(model.layers[-2].input, model.layers[-1].output)

    patch_size = 80 #changed from 192
    craft = Craft(input_to_latent = g,
                latent_to_logit = h,
                number_of_concepts = 10,
                patch_size = patch_size,
                batch_size = 64,
               )

   
    activations, patches = craft.fit(images_preprocessed)
    crops, crops_u, w = craft.activation_transform(activations, patches)
    importances = craft.estimate_importance(
            images_preprocessed, class_labels=labels
        )
    most_important_concepts = helpers.plot_new_histogram(
        importances, histogram_dir
    )
    helpers.save_new_crops(
        most_important_concepts,
        importances,
        crops_u,
        crops,
        save_crops,
        0
    )


if __name__ == "__main__":
    
    main()
