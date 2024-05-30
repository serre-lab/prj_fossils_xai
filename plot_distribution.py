import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
import numpy as np
import matplotlib.pyplot as plt
from Craft.craft.new_craft_tf import Craft
import helpers

def plot_histogram(data, bins=20, title="Histogram", xlabel="Values", ylabel="Frequency", save_path=None):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # Calculate percentiles
    percentiles = np.percentile(data, [25, 50, 75])
    
    # Draw percentile lines
    for percentile in percentiles:
        plt.axvline(percentile, color='r', linestyle='--', linewidth=1)
        plt.text(percentile, plt.ylim()[1]*0.9, f'{int(percentile)}th', color='r')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def evaluate():
    model_path = "./models/model-13.h5"
    model,g,h = helpers.get_model(model_path)
    activations_and_patches = np.load(
        "./activations/activations_patches.npz"
    )
    activations = activations_and_patches["activations"]
    patches = activations_and_patches["patches"]

    indices = np.arange(activations.shape[0])
    np.random.shuffle(indices)
    activations = tf.gather(activations, indices)
    patches = tf.gather(patches, indices)
    # final_labels = final_labels[indices]
    patch_size = 192
    craft = Craft(
        input_to_latent=g,
        latent_to_logit=h,
        number_of_concepts=20,
        patch_size=patch_size,
        batch_size=64,
    )
    crops, crops_u, w = craft.activation_transform(activations, patches)
    x = crops_u[:, 1]
    print(x[::-1].numpy())
    plot_histogram(x, bins=5, save_path="histogram_percentile.png")



    


if __name__ == "__main__":
    evaluate()
