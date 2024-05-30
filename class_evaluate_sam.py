import torch
# torch.cuda.set_per_process_memory_fraction(0.2, device=0)
print('segment anything')
from segment_anything import SamPredictor, sam_model_registry
print('importing')
sam = sam_model_registry["default"]("./models/sam_02-06_dice.pth")
sam.cuda()
print("defining")
predictor = SamPredictor(sam)

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
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
#import segmentation_models as sm
import json
#sm.set_framework('tf.keras')
#sm.framework()
import cv2
import helpers



def prepare_dataset(cid, x_tests, is_leaves=True):
    images = []
    images_anchor1 = []
    images_anchor2 = []
    labels = []
    
    count = 0
    if is_leaves:
        images, labels, domain = load_leaves(cid, x_tests)
        print(f"Length of images : {len(images)}")
        print(f"Sam Processing >>>>>>>>>>>")
        # ipdb.set_trace()
        images, images_zoom, labels, _ = segmentation_sam(
            images, labels, domain, sam, predictor
        )
        # images = samples[0]
        # print(images.shape)
        # images_zoom = samples[3]
        print(f"<<<<<<<<<< Processing Done")
        # ipdb.set_trace()
        images = images / 255
        images_zoom = images_zoom / 255
    else:
        if cid in x_tests:
            dataset = x_tests[cid]
            for ele in dataset:
                try:
                    im, label = parse_fossils(ele, False, False)
                    images.append(im)
                    labels.append(label)
                except Exception as error:
                    print(f"Error : {count}")
                    count += 1
                    continue
        else:
            print(f"The class {cid} donot have fossils samples")
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    images_zoom = tf.convert_to_tensor(images_zoom, dtype=tf.float32)
    # images_zoom2 = tf.convert_to_tensor(images_zoom2, dtype = tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    print(f"Failed Extraction : {count}")
    print(type(images), images.shape)
    print(type(images_zoom), images_zoom.shape)
    # print(type(images_zoom2), images_zoom2.shape)
    print(type(labels), labels.shape)
    return images, images_zoom, labels


classes_to_evaluate = [1]
# def main(x_fossils, x_leaves):

# class_leaves_accuracy = []
class_fossils_accuracy = []
for class_id in classes_to_evaluate:

  class_id = int(class_id)

  # images_leaves, images_leaves_zoom, labels = prepare_dataset(cid, x_leaves, True)
  images_fossils, _, labels = helpers.prepare_dataset(cid, x_fossils, False)

  # leaves_predictions  = top5_predictions(images_leaves_zoom, cid)
  fossils_predictions, latents = top5_predictions(images_fossils, cid)


  y_true.extend([cid for i in range(len(images_fossils))])
  y_pred.extend(fossils_predictions)

  # images_leaves_correct  = images_leaves_zoom[leaves_predictions == cid]
  images_fossils_correct = images_fossils[fossils_predictions == cid]

  # leaves_accuracy = len(images_leaves_correct)/len(images_leaves)
  fossils_accuracy = len(images_fossils_correct)/len(images_fossils)

  # print(f'Class Leaves {cid} Accuracy : {leaves_accuracy}')
  print(f'Class Fossils {cid} Accuracy : {fossils_accuracy}')
  # class_leaves_accuracy.append(leaves_accuracy)
  class_fossils_accuracy.append(fossils_accuracy)


  # if len(images_leaves_correct) == 0:
  #   print(f'Class {cid} : {class_names[int(cid)]} has zero correct leaves samples')
  #   continue
  if len(images_fossils_correct) == 0:
    print(f'Class {cid} : {id_to_class[cid]} has zero correct fossils samples')
    continue
  else:
    print(f'Class {cid} : {id_to_class[cid]} has {len(images_fossils_correct)} correct fossils samples out of {len(images_fossils)}')
  # print(f'Class {cid} : {class_names[int(cid)]} has {len(images_leaves_correct)} correct leaves samples out of {len(images_leaves)}

  # if images_leaves_correct.shape[0]<= images_fossils_correct.shape[0]:
  #   images_fossils_correct = images_fossils_correct[:images_leaves_correct.shape[0]]
  # else:
  #   images_leaves_correct = images_leaves_correct[:images_fossils_correct.shape[0]]

  # assert images_leaves_correct.shape == images_fossils_correct.shape

  # final_images = tf.concat([images_leaves_correct, images_fossils_correct], 0)

  # # return images_leaves_correct, images_fossils_correct, final_images
  # print(images_leaves_correct.shape, images_fossils_correct.shape, final_images.shape)

  start = time.time()
  craft = Craft(input_to_latent = g,
                latent_to_logit = h,
                number_of_concepts = 20,
                patch_size = 96,
                batch_size = 32)
  crops, crops_u, w = craft.fit(images_fossils_correct)
  end = time.time()
  print(end - start)
  print(f'crops shape: {crops.shape}, crops_u shape: {crops_u.shape}, w shape: {w.shape}')
  importances = craft.estimate_importance(images_fossils_correct, class_id=cid) # 330 is the rabbit class id in imagenet
  images_u = craft.transform(images_fossils_correct)

  most_important_concepts = plot_histogram(importances, cid)
  save_crops(most_important_concepts, importances, crops_u, crops, cid)