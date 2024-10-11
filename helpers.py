import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpu_devices[0], True)


import keras
from keras_cv_attention_models import beit


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# ipdb.set_trace()

# from leaves_processing import load_leaves
# from sam_preprocessing import (
#     segmentation_sam,
#     get_segmentation_model,
#     segmentation_augmentation,
# )
# from fossils_processing import parse_fossils

# import ipdb

class_names = {
    "Fabaceae": 5,
    "Annonaceae": 26,
    "Lauraceae": 8,
    "Rubiaceae": 116,
    "Euphorbiaceae": 63,
    "Thymelaeaceae": 133,
    "Polygalaceae": 110,
    "Rhamnaceae": 12,
    "Fagaceae": 6,
    "Sapindaceae": 15,
    "Sapotaceae": 120,
    "Apiaceae": 27,
    "Berberidaceae": 1,
    "Ericaceae": 61,
    "Theaceae": 132,
    "Lardizabalaceae": 75,
    "Malvaceae": 83,
    "Acanthaceae": 21,
    "Rosaceae": 13,
    "Anacardiaceae": 0,
    "Cornaceae": 52,
    "Myristicaceae": 90,
    "Meliaceae": 9,
    "Hamamelidaceae": 68,
    "Cupressaceae": 3,
    "Ebenaceae": 58,
    "Myrtaceae": 10,
    "Apocynaceae": 28,
    "Salicaceae": 14,
    "Passifloraceae": 101,
    "Ranunculaceae": 114,
    "Celastraceae": 45,
    "Ulmaceae": 16,
    "Pittosporaceae": 107,
    "Asteraceae": 32,
    "Piperaceae": 106,
    "Betulaceae": 2,
    "Juglandaceae": 7,
    "Winteraceae": 138,
    "Menispermaceae": 86,
    "Cunoniaceae": 55,
    "Rutaceae": 117,
    "Combretaceae": 49,
    "Elaeocarpaceae": 60,
    "Oleaceae": 96,
    "Proteaceae": 113,
    "Capparaceae": 42,
    "Aquifoliaceae": 29,
    "Gnetaceae": 67,
    "Phyllanthaceae": 104,
    "Bignoniaceae": 33,
    "Melastomataceae": 85,
    "Styracaceae": 130,
    "Rhizophoraceae": 115,
    "Lecythidaceae": 76,
    "Ochnaceae": 94,
    "Schisandraceae": 123,
    "Chloranthaceae": 46,
    "Lamiaceae": 74,
    "Onagraceae": 97,
    "Araliaceae": 30,
    "Moraceae": 88,
    "Malpighiaceae": 82,
    "Caprifoliaceae": 43,
    "Icacinaceae": 71,
    "Lythraceae": 80,
    "Saxifragaceae": 122,
    "Pinaceae": 11,
    "Chrysobalanaceae": 47,
    "Dilleniaceae": 56,
    "Calophyllaceae": 37,
    "Loranthaceae": 79,
    "Penaeaceae": 102,
    "Polemoniaceae": 109,
    "Actinidiaceae": 23,
    "Magnoliaceae": 81,
    "Buxaceae": 36,
    "Dryopteridaceae": 4,
    "Connaraceae": 50,
    "Crassulaceae": 53,
    "Hypericaceae": 70,
    "Urticaceae": 134,
    "Nothofagaceae": 91,
    "Nyssaceae": 93,
    "Achariaceae": 22,
    "Burseraceae": 35,
    "Symplocaceae": 131,
    "Loganiaceae": 78,
    "Dipterocarpaceae": 57,
    "Garryaceae": 64,
    "Campanulaceae": 39,
    "Vitaceae": 18,
    "Viburnaceae": 17,
    "Humiriaceae": 69,
    "Sabiaceae": 118,
    "Altingiaceae": 24,
    "Pentaphylacaceae": 103,
    "Polygonaceae": 111,
    "Simaroubaceae": 125,
    "Cardiopteridaceae": 44,
    "Violaceae": 136,
    "Coriariaceae": 51,
    "Platanaceae": 108,
    "Amaranthaceae": 25,
    "Geraniaceae": 65,
    "Monimiaceae": 87,
    "Santalaceae": 119,
    "Olacaceae": 95,
    "Iteaceae": 72,
    "Cannabaceae": 41,
    "Linaceae": 77,
    "Clusiaceae": 48,
    "Cucurbitaceae": 54,
    "Elaeagnaceae": 59,
    "Zygophyllaceae": 139,
    "Boraginaceae": 34,
    "Stemonuraceae": 129,
    "Gesneriaceae": 66,
    "Escalloniaceae": 62,
    "Phytolaccaceae": 105,
    "Oxalidaceae": 99,
    "Sarcolaenaceae": 121,
    "Verbenaceae": 135,
    "Staphyleaceae": 128,
    "Canellaceae": 40,
    "Aristolochiaceae": 31,
    "Myricaceae": 89,
    "Primulaceae": 112,
    "Marantaceae": 84,
    "Paracryphiaceae": 100,
    "Scrophulariaceae": 124,
    "Solanaceae": 127,
    "Smilacaceae": 126,
    "Ixonanthaceae": 73,
    "Nyctaginaceae": 92,
    "Vochysiaceae": 137,
    "Calycanthaceae": 38,
    "Opiliaceae": 98,
    "Taxaceae": 22,
    "Araceae": 19,
}


def get_model(model_path):
    # backbone = beit.BeitBasePatch16(input_shape = (384,384,3), pretrained = "imagenet21k-ft1k")
    cce = tf.keras.losses.categorical_crossentropy
    model = keras.models.load_model(model_path, custom_objects={"cce": cce})
    g = tf.keras.Model(model.input, model.layers[-3].output)
    # out = tf.keras.layers.Activation('relu')(g_.output)
    # g = tf.keras.Model(model.input, out)
    h = tf.keras.Model(model.layers[-1].input, model.layers[-1].output)

    return model, g, h


def get_resnet(
    base_arch="Nasnet", weights="imagenet", input_shape=(600, 600, 3), classes=64500
):
    if base_arch == "Nasnet":
        base_model = tf.keras.applications.NASNetLarge(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            input_tensor=None,
            pooling=None,
        )
    elif base_arch == "Resnet50v2":
        base_model = tf.keras.applications.ResNet50V2(
            weights=weights, include_top=False, pooling="avg", input_shape=input_shape
        )
    elif base_arch == "Resnet101":
        base_model = tf.keras.applications.ResNet101(
            weights=weights, include_top=False, pooling="avg", input_shape=input_shape
        )
    #   import ipdb;ipdb.set_trace()
    model = tf.keras.Sequential(
        [base_model, tf.keras.layers.Dense(classes, activation="softmax")]
    )

    return base_model, model


def get_resnet_model(model_path):
    # import ipdb;ipdb.set_trace()
    cce = tf.keras.losses.categorical_crossentropy
    model = keras.models.load_model(model_path, custom_objects={"cce": cce})
    g = keras.Model(model.input, model.layers[-3].output)
    # out = tf.keras.layers.Activation('relu')(g_.output)
    # g = tf.keras.Model(model.input, out)
    h = keras.Model(model.layers[-1].input, model.layers[-1].output)
    return model, g, h


def get_families(csv_path):
    df = pd.read_csv(csv_path, index_col=0)

    families = df["family"].unique()
    ignore_families = []

    for fam in families:
        fam_df = df[df["family"] == fam]
        unique_classes = fam_df["label"].unique()
        if len(unique_classes) > 1:
            print(f"Multiple class in family {fam} : {fam_df['label'].unique()}")
            ignore_families.append(fam)

    if len(ignore_families) > 0:
        df = df[~(df["family"].isin(ignore_families))]

    families = df["family"].unique()
    n_classes = len(families)
    print(f"Total Number of families : {n_classes}")

    class_names = []
    class_labels = []

    for fam in families:
        fam_df = df[df["family"] == fam]
        unique_classes = fam_df["label"].unique()
        class_names.append(fam)
        class_labels.append(unique_classes[0])

    return class_names, class_labels


def prepare_dataset(cid, x_tests, is_leaves=True):
    ## prepare dataset

    if is_leaves:
        sam, predictor = get_segmentation_model()

    images = []
    images_zoom = []
    # images_zoom2 = []
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


LR = 1e-3


def top5_predictions(model, images, cid):
    outputs = model.predict(images)
    for op in outputs:
        print(op.shape)
    predictions = tf.math.top_k(outputs[1], k=5)
    final_predictions = []
    for ele in predictions[1]:
        if cid in ele:
            final_predictions.append(cid)
        else:
            final_predictions.append(ele[0])
    final_predictions = np.array(final_predictions)
    return final_predictions, outputs[0]


def plot_samples(images):
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    # Plot the images
    for i in range(5):
        for j in range(5):
            index = i * 5 + j
            axes[i, j].imshow(images[index])
            axes[i, j].axis("off")
    # Turn off axis for the entire figure
    plt.axis("off")
    plt.show()


def plot_histogram(importances, cid, class_names, histogram_dir):
    if not os.path.exists(histogram_dir):
        os.makedirs(histogram_dir)
    plt.bar(range(len(importances)), importances)
    plt.xticks(range(len(importances)))
    plt.title("Concept Importance")
    plt.savefig(histogram_dir + f"/{class_names[cid]}_{cid}.png")
    plt.show()
    most_important_concepts = np.argsort(importances)[::-1][:5]
    for c_id in most_important_concepts:
        print("Concept", c_id, " has an importance value of ", importances[c_id])
    return most_important_concepts


def plot_new_histogram(importances, histogram_dir, b, n_concepts=10):
    if not os.path.exists(histogram_dir):
        os.makedirs(histogram_dir)
    plt.figure(figsize=(20, 10))
    plt.bar(range(len(importances)), importances)
    # plt.xticks(range(len(importances)))
    plt.title("Concept Importance")
    plt.savefig(histogram_dir + f"/all_class_histogram_till_{b}.png")
    plt.show()
    most_important_concepts = np.argsort(importances)[::-1][:n_concepts]
    # for c_id in most_important_concepts:
    #     print("Concept", c_id, " has an importance value of ", importances[c_id])
    return most_important_concepts


from math import ceil


def save_crops(
    most_important_concepts, importances, crops_u, crops, cid, id_to_class, save_dir
):
    nb_crops = 10

    def show(img):
        img = np.array(img)
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        return

    for c_id in most_important_concepts:
        if not os.path.exists(
            save_dir
            + f"/{id_to_class[cid]}_{cid}/concept_{c_id}_importance_{importances[c_id]}"
        ):
            os.makedirs(
                save_dir
                + f"/{id_to_class[cid]}_{cid}/concept_{c_id}_importance_{importances[c_id]}"
            )
        best_crops_ids = np.argsort(crops_u[:, c_id])[::-1]
        best_crops = np.array(crops)[best_crops_ids]
        print("Concept", c_id, " has an importance value of ", importances[c_id])

        niter = crops_u.shape[0] // nb_crops
        niter = 10 if niter > 10 else niter
        for j in range(niter):
            plt.figure(figsize=(10, 5))
            for i in range(nb_crops):
                plt.subplot(ceil(nb_crops / 5), 5, i + 1)
                show(best_crops[j * nb_crops + i])
            plt.tight_layout()
            plt.savefig(
                save_dir
                + f"/{id_to_class[cid]}_{cid}/concept_{c_id}_importance_{importances[c_id]}/{j}.png"
            )
            plt.show()
            print("\n\n")


def save_classwise_crops(
    most_important_concepts,
    importances,
    crops_u,
    crops,
    crops_labels,
    save_dir,
    batch,
    nb_crops,
):
    for c_id in most_important_concepts:
        if importances[c_id] == 0.0:
            print("Terminating....! concept importance 0.0 found")
            break

        output_dir = os.path.join(
            save_dir,
            f"all_images_v2_till_batch_{batch}/concept_{c_id}_importance_{importances[c_id]}",
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )
        best_crops_ids = np.argsort(crops_u[:, c_id])[::-1]
        best_crops_labels = np.array(crops_labels)[best_crops_ids]
        best_crops = np.array(crops)[best_crops_ids]
        print("Concept", c_id, " has an importance value of ", importances[c_id])

        for cls_name, idx in class_names.items():
            best_class_crops = best_crops[best_crops_labels == idx]
            if len(best_class_crops) == 0:
                continue
            class_dir = os.path.join(output_dir, f"{cls_name}-{idx}")
            os.makedirs(
                class_dir,
                exist_ok=True,
            )
            niter = best_class_crops.shape[0] // nb_crops
            niter = 50 if niter > 50 else niter
            x = nb_crops // 5

            if niter == 0:
                canvas_height = (
                    x * 128
                )  # Assuming each crop is roughly 100 pixels in height
                canvas_width = (
                    5 * 128
                )  # Assuming each crop is roughly 100 pixels in width
                canvas = np.ones((canvas_width, canvas_height, 3)) * 255

                for i in range(nb_crops):
                    row = i // 5
                    col = i % 5
                    if i == best_class_crops.shape[0]:
                        break
                    crop = best_class_crops[i]
                    crop_height, crop_width, _ = crop.shape
                    start_y = row * 128
                    start_x = col * 128
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    canvas[
                        start_y : start_y + crop_height, start_x : start_x + crop_width
                    ] = (crop * 255)
                # Save the image using OpenCV
                # import ipdb;ipdb.set_trace()
                output_path = os.path.join(class_dir, f"{0}.png")
                cv2.imwrite(output_path, canvas)

            for j in range(niter):
                canvas_height = (
                    x * 128
                )  # Assuming each crop is roughly 100 pixels in height
                canvas_width = (
                    5 * 128
                )  # Assuming each crop is roughly 100 pixels in width
                canvas = np.ones((canvas_width, canvas_height, 3)) * 255

                for i in range(nb_crops):
                    row = i // 5
                    col = i % 5
                    crop = best_class_crops[j * nb_crops + i]
                    crop_height, crop_width, _ = crop.shape
                    start_y = row * 128
                    start_x = col * 128
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    canvas[
                        start_y : start_y + crop_height, start_x : start_x + crop_width
                    ] = (crop * 255)
                # Save the image using OpenCV
                # import ipdb;ipdb.set_trace()
                output_path = os.path.join(class_dir, f"{j}.png")
                cv2.imwrite(output_path, canvas)


def save_new_crops(
    most_important_concepts,
    importances,
    crops_u,
    crops,
    crops_labels,
    save_dir,
    batch,
    nb_crops,
):
    for c_id in most_important_concepts:
        if importances[c_id] == 0.0:
            print("Terminating....! concept importance 0.0 found")
            break

        output_dir = os.path.join(
            save_dir,
            f"all_images_v2_till_batch_{batch}/concept_{c_id}_importance_{importances[c_id]}",
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )
        best_crops_ids = np.argsort(crops_u[:, c_id])[::-1]
        best_crops = np.array(crops)[best_crops_ids]
        print("Concept", c_id, " has an importance value of ", importances[c_id])

        niter = crops_u.shape[0] // nb_crops
        niter = 50 if niter > 50 else niter
        x = nb_crops // 5

        for j in range(niter):
            canvas_height = (
                x * 128
            )  # Assuming each crop is roughly 100 pixels in height
            canvas_width = 5 * 128  # Assuming each crop is roughly 100 pixels in width
            canvas = np.zeros((canvas_width, canvas_height, 3))

            for i in range(nb_crops):
                row = i // 5
                col = i % 5
                crop = best_crops[j * nb_crops + i]
                crop_height, crop_width, _ = crop.shape
                start_y = row * 128
                start_x = col * 128
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                canvas[
                    start_y : start_y + crop_height, start_x : start_x + crop_width
                ] = (crop * 255)
            # Save the image using OpenCV
            # import ipdb;ipdb.set_trace()
            output_path = os.path.join(output_dir, f"{j}.png")
            cv2.imwrite(output_path, canvas)

        # for j in range(niter):
        #     best_crops_norm = [
        #         new_show(best_crops[i])
        #         for i in range(j * nb_crops, j * nb_crops + 10, 1)
        #     ]
        #     rows = [
        #         np.concatenate(best_crops_norm[i : i + 5], axis=1)
        #         for i in range(j * nb_crops, j * nb_crops + 10, 5)
        #     ]
        #     final_image = np.concatenate(rows, axis=0)
        #     final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(
        #         save_dir
        #         + f"/all_images_v2_till_batch_{batch}/concept_{c_id}_importance_{importances[c_id]}/{j}.png",
        #         final_image,
        #     )
