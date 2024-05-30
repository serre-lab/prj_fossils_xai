import time
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from craft.craft_tf import Craft
from leaves_processing import get_leaves_paths
from fossils_processing import get_fossils_paths
import helpers

import ipdb


def main():
    model_path = "./models/model-25.h5"
    csv_path = "./csv/fossils.csv"
    fossils_data_dir = (
        "/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Florissant_Fossil_v2.0"
    )
    leaves_data_dir = (
        "/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Extant_Leaves"
    )
    save_crops = "./crops/fossils_leaves_crops/exp5_RELU_192_20"
    histogram_dir = "./histogram/exp5_RELU_192_20"

    common_class = [1, 0, 5]

    class_leaves_accuracy = []
    class_fossils_accuracy = []
    class_zoom_leaves_accuracy = []

    class_names, class_labels = helpers.get_families(csv_path)
    print("get_families passed!!!")

    class_to_id = {c: class_labels[i] for i, c in enumerate(class_names)}
    id_to_class = {i: c for c, i in class_to_id.items()}

    x_leaves = get_leaves_paths(class_names, class_to_id, id_to_class, leaves_data_dir)
    print("get_leaves_path passed!!!")
    x_fossils = get_fossils_paths(
        class_names, class_to_id, id_to_class, fossils_data_dir
    )
    print("get_fossils_path passed!!!")

    model, g, h = helpers.get_model(model_path)
    # import ipdb

    # ipdb.set_trace()
    print("get_model passed !!!")

    for cid in common_class:
        cid = int(cid)
        # ipdb.set_trace()
        images_leaves, images_leaves_zoom, labels = helpers.prepare_dataset(
            cid, x_leaves, True
        )
        images_fossils, _, labels = helpers.prepare_dataset(cid, x_fossils, False)
        print("prepare_dataset passed!!!")

        # ipdb.set_trace()
        if len(images_leaves) > 0:
            zoom_leaves_predictions, latents = helpers.top5_predictions(
                model, images_leaves_zoom, cid
            )
            leaves_predictions, latents = helpers.top5_predictions(
                model, images_leaves, cid
            )
            images_leaves_correct = images_leaves[leaves_predictions == cid]
            images_leaves_zoom_correct = images_leaves_zoom[
                zoom_leaves_predictions == cid
            ]

            leaves_accuracy = len(images_leaves_correct) / len(images_leaves)
            zoom_leaves_accuracy = len(images_leaves_zoom_correct) / len(
                images_leaves_zoom
            )

        if len(images_fossils) > 0:
            fossils_predictions, latents = helpers.top5_predictions(
                model, images_fossils, cid
            )
            images_fossils_correct = images_fossils[fossils_predictions == cid]
            fossils_accuracy = len(images_fossils_correct) / len(images_fossils)
        else:
            fossils_accuracy = "No samples"

        # y_true.extend([cid for i in range(len(images_fossils))])
        # y_pred.extend(fossils_predictions)

        print(f"Class Leaves {cid} Accuracy : {leaves_accuracy}")
        print(f"Class Fossils {cid} Accuracy : {fossils_accuracy}")
        print(f"Class Zoom Leaves {cid} Accuracy : {zoom_leaves_accuracy}")

        class_leaves_accuracy.append(leaves_accuracy)
        class_fossils_accuracy.append(fossils_accuracy)
        class_zoom_leaves_accuracy.append(zoom_leaves_accuracy)

        if len(images_leaves_correct) == 0:
            print(
                f"Class {cid} : {id_to_class[int(cid)]} has zero correct leaves samples"
            )
        else:
            print(
                f"Class {cid} : {id_to_class[int(cid)]} has {len(images_leaves_correct)} correct leaves samples out of {len(images_leaves)}"
            )

        if len(images_fossils_correct) == 0:
            print(f"Class {cid} : {id_to_class[cid]} has zero correct fossils samples")
            # continue
        else:
            print(
                f"Class {cid} : {id_to_class[cid]} has {len(images_fossils_correct)} correct fossils samples out of {len(images_fossils)}"
            )

        # if images_leaves_correct.shape[0]<= images_fossils_correct.shape[0]:
        #   images_fossils_correct = images_fossils_correct[:images_leaves_correct.shape[0]]
        # else:
        #   images_leaves_correct = images_leaves_correct[:images_fossils_correct.shape[0]]

        # assert images_leaves_correct.shape == images_fossils_correct.shape

        final_images = tf.concat([images_leaves_correct, images_fossils_correct], 0)

        # # return images_leaves_correct, images_fossils_correct, final_images
        # print(images_leaves_correct.shape, images_fossils_correct.shape, final_images.shape)

        start = time.time()
        craft = Craft(
            input_to_latent=g,
            latent_to_logit=h,
            number_of_concepts=20,
            patch_size=192,
            batch_size=64,
        )
        crops, crops_u, w = craft.fit(final_images)
        end = time.time()
        print(f"time required by craft - {end - start}")
        print(
            f"crops shape: {crops.shape}, crops_u shape: {crops_u.shape}, w shape: {w.shape}"
        )
        importances = craft.estimate_importance(
            final_images, class_id=cid
        )  # 330 is the rabbit class id in imagenet
        images_u = craft.transform(images_fossils_correct)

        most_important_concepts = helpers.plot_histogram(
            importances, cid, id_to_class, histogram_dir
        )
        helpers.save_crops(
            most_important_concepts,
            importances,
            crops_u,
            crops,
            cid,
            id_to_class,
            save_crops,
        )


if __name__ == "__main__":
    main()
