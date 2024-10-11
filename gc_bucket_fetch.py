# import tensorflow as tf
import os
from google.cloud import storage
import cv2
import numpy as np
import tensorflow as tf

## load data
def  download_folder_from_gcs(bucket_name, gcs_folder_path, local_folder_path, class_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix = gcs_folder_path)
    for blob in blobs:
        curr_class_name, file_name = blob.name.split('/')[-2:]
        print(curr_class_name, file_name)
        if not file_name or curr_class_name!=class_name:
            continue
        local_file_path = os.path.join(local_folder_path, file_name)
        blob.download_to_filename(local_file_path)
        print("Download compelete")

## load model
def get_model(model_path):
    cce = tf.keras.losses.categorical_crossentropy
    model = tf.keras.models.load_model(model_path, custom_objects= {'cce': cce})
    print(model.summary())
    features = tf.keras.Model(model.input, model.layers[-5].output)
    return model, features

## Preprocessing 
def load_images(paths):
    fossils = []
    count = 0
    for fossils_path in paths:
        img = cv2.imread(fossils_path)[...,::-1]
        img = img.astype(np.float32)
        fossils.append(img)
        count+=1
    print(f"Total fossils : {count}")    
    return fossils
        
if __name__ == "__main__":
    bucket_name = "serrelab"
    gcs_folder_path = "prj_fossils/2024/Extant_Leaves"   # ("Extant_Leaves" or "Florissant_Fossil_v2.0")
    local_folder_path = "./trash"
    class_name = "Anacardiaceae"
    local_folder_path = os.path.join(local_folder_path, class_name)
    os.makedirs(local_folder_path, exist_ok=True)
    
    download_folder_from_gcs(bucket_name, gcs_folder_path, local_folder_path, class_name)

    image_paths = [os.path.join("./trash", class_name, img) for img in os.listdir('./trash/' + class_name)]
    X = load_images(image_paths)
    resized_images = [tf.image.resize(tf.convert_to_tensor(img), (384, 384)) for img in X]
    resized_images = tf.reshape(resized_images, (-1, 384, 384, 3))/255.0
    
    model_path = ''
    model = get_model(model_path)
    _, batch_logits = model.predict(resized_images)