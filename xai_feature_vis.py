import tensorflow as tf
import numpy as np
from tqdm import tqdm

import xplique
from xplique.features_visualizations import Objective
from xplique.features_visualizations import maco
from xplique.plot import plot_maco


def to_numpy(tensor):
    # Ensure tensor is on CPU and convert to NumPy
    return np.array(tensor).astype(np.float32)


def check_format(arr):
    # ensure numpy array and move channels to the last dimension
    # if they are in the first dimension
    arr = to_numpy(arr)
    if arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1)
    return arr


def normalize(image):
    # normalize image to 0-1 range
    image = np.array(image, dtype=np.float32)
    image -= image.min()
    image /= image.max()
    return image


def standardize_np(image):
    # normalize image to 0-1 range
    image = np.array(image, dtype=np.float32)
    image -= image.mean()
    image /= image.std() + 1e-3
    return image


def clip_percentile(img, p=0.1):
    # clip pixel values to specified percentile range
    return np.clip(img, np.percentile(img, p), np.percentile(img, 100 - p))


def show(img, **kwargs):
    # display image with normalization and channels in the last dimension
    img = check_format(img)
    img = normalize(img)

    plt.imshow(img, **kwargs)
    plt.axis("off")
    # plt.show()


def plot_maco(image, alpha, percentile_image=0.5, percentile_alpha=85):
    # visualize image with alpha mask overlay after normalization and clipping
    image, alpha = check_format(image), check_format(alpha)
    image = standardize_np(image)
    image = normalize(image)
    image = clip_percentile(image, percentile_image)

    # mean of alpha across channels, clipping, and normalization
    alpha = np.mean(alpha, -1, keepdims=True)
    alpha = np.clip(alpha, None, np.percentile(alpha, percentile_alpha))
    alpha = alpha / alpha.max()

    # image = image * alpha

    # overlay alpha mask on the image
    plt.imshow(np.concatenate([image, alpha], -1))
    plt.axis("off")
    # plt.show()


def cosine_similarity(tensor_a, tensor_b):
    # Calculate cosine similarity
    norm_dims = list(range(1, len(tensor_a.shape)))
    tensor_a = tf.math.l2_normalize(tensor_a, axis=norm_dims)
    tensor_b = tf.math.l2_normalize(tensor_b, axis=norm_dims)
    return tf.reduce_sum(tensor_a * tensor_b, axis=norm_dims)


def dot_cossim(tensor_a, tensor_b, cossim_pow=2.0):
    # Compute dot product scaled by cosine similarity
    cosim = tf.math.pow(
        tf.clip_by_value(cosine_similarity(tensor_a, tensor_b), 1e-1, 1.0), cossim_pow
    )
    dot = tf.reduce_sum(tensor_a * tensor_b)
    return dot * cosim


# tensor for color correlation svd square root
color_correlation_svd_sqrt = tf.constant(
    [
        [0.56282854, 0.58447580, 0.58447580],
        [0.19482528, 0.00000000, -0.19482528],
        [0.04329450, -0.10823626, 0.06494176],
    ],
    dtype=tf.float32,
)


def standardize(tensor):
    # standardizes the tensor to have 0 mean and unit variance
    tensor = tensor - tf.reduce_mean(tensor)
    tensor = tensor / (tf.math.reduce_std(tensor) + 1e-4)
    return tensor


def recorrelate_colors(image):
    # recorrelates the colors of the images
    assert len(image.shape) == 3
    assert image.shape[-1] == 3

    flat_image = tf.reshape(image, [-1, 3])

    recorrelated_image = tf.matmul(flat_image, color_correlation_svd_sqrt)
    recorrelated_image = tf.reshape(recorrelated_image, image.shape)

    return recorrelated_image


def batch_half_grayscale(images):
    batch_size = tf.shape(images)[0]
    mid_point = batch_size // 2
    grayscale_images = tf.image.rgb_to_grayscale(images[:mid_point])
    grayscale_images = tf.tile(grayscale_images, [1, 1, 1, 3])
    output_images = tf.concat([grayscale_images, images[mid_point:]], axis=0)
    return output_images


@tf.function
def optimization_step(
    objective_function,
    image,
    box_size,
    noise_level,
    number_of_crops_per_iteration,
    model_input_size,
):
    # performs an optimization step on the generated image
    assert box_size[1] >= box_size[0]
    assert len(image.shape) == 3
    assert image.shape[-1] == 3

    # generate random boxes
    x0 = 0.5 + tf.random.normal((number_of_crops_per_iteration,)) * 0.15
    y0 = 0.5 + tf.random.normal((number_of_crops_per_iteration,)) * 0.15
    delta_x = (
        tf.random.uniform((number_of_crops_per_iteration,))
        * (box_size[1] - box_size[0])
        + box_size[0]
    )
    delta_y = delta_x

    box_indices = tf.zeros(shape=(number_of_crops_per_iteration,), dtype=tf.int32)
    boxes = tf.stack(
        [
            x0 - delta_x * 0.5,
            y0 - delta_y * 0.5,
            x0 + delta_x * 0.5,
            y0 + delta_y * 0.5,
        ],
        -1,
    )

    crops = tf.image.crop_and_resize(
        image[None, :, :, :], boxes, box_indices, (model_input_size, model_input_size)
    )

    # Add normal and uniform noise for better robustness
    # crops += tf.random.normal(tf.shape(crops)) * noise_level
    # crops += (tf.random.uniform(tf.shape(crops)) - 0.5) * noise_level
    # color noise
    # crops += tf.random.normal((3,))[None, None, None, :] * noise_level
    # crops = batch_half_grayscale(crops)

    # compute the score and loss
    score = objective_function(crops)
    loss = -score

    return loss, image


def fft_2d_freq(width: int, height: int) -> np.ndarray:
    freq_y = np.fft.fftfreq(height)[:, np.newaxis].astype(np.float64)

    cut_off = int(width % 2 == 1)
    freq_x = np.fft.fftfreq(width)[: width // 2 + 1 + cut_off]

    return np.sqrt(freq_x**2 + freq_y**2)


def get_fft_scale(width: int, height: int, decay_power: float = 1.0) -> tf.Tensor:
    frequencies = fft_2d_freq(width, height)
    fft_scale = 1.0 / np.maximum(frequencies, 1.0 / max(width, height)) ** decay_power
    fft_scale = fft_scale * np.sqrt(width * height)

    return tf.cast(fft_scale, dtype=tf.complex64)


def init_olah_buffer(width, height, std=1e-3):
    # Initialize the Olah buffer with a random spectrum
    spectrum_shape = (3, width, height // 2 + 1)
    random_spectrum = tf.complex(
        tf.random.normal(spectrum_shape) * std, tf.random.normal(spectrum_shape) * std
    )
    return random_spectrum


def fourier_preconditioner(spectrum, spectrum_scaler, values_range):
    # Precondition the Fourier spectrum and convert it to spatial domain
    assert spectrum.shape[0] == 3

    # spectrum = standardize_complex(spectrum)
    spectrum = spectrum * spectrum_scaler

    spatial_image = tf.signal.irfft2d(spectrum)
    spatial_image = tf.transpose(spatial_image, [1, 2, 0])

    image = spatial_image
    image = standardize(image) / 2.0
    image = recorrelate_colors(image)
    # color_recorrelated_image = spatial_image
    # image = spatial_image
    # image = color_recorrelated_image

    image = tf.nn.sigmoid(image)
    # image = image - tf.reduce_min(image)
    # image = image / (tf.reduce_max(image) + 1e-3)
    image = image * (values_range[1] - values_range[0]) + values_range[0]

    # image = tf.sigmoid(image) * (values_range[1] - values_range[0]) + values_range[0]

    # mean = tf.reduce_mean(image, (0, 1))
    # image = image - (image - mean[None, None, :]) * 0.5
    # image = image * (values_range[1] - values_range[0]) + values_range[0]
    return image


def fourier(
    objective_function,
    decay_power=1.5,
    total_steps=1000,
    learning_rate=1.0,
    image_size=1280,
    model_input_size=384,
    noise=0.08,
    values_range=(-0.1, 1.1),
    crops_per_iteration=8,
    box_size=(0.15, 0.25),
    device="/GPU:0",
):
    # Perform the Olah optimization process
    assert values_range[1] >= values_range[0]
    assert box_size[1] >= box_size[0]

    spectrum = init_olah_buffer(image_size, image_size, std=1.0)
    spectrum_scaler = get_fft_scale(image_size, image_size, decay_power)

    with tf.device(device):
        spectrum = tf.Variable(spectrum)
        optimizer = tf.optimizers.Nadam(learning_rate=learning_rate)

        transparency_accumulator = tf.zeros(
            (image_size, image_size, 3), dtype=tf.float32
        )

        @tf.function
        def sstep(spectrum):
            with tf.GradientTape() as tape:
                tape.watch(spectrum)
                image = fourier_preconditioner(spectrum, spectrum_scaler, values_range)
                tape.watch(image)
                # set_size(1, 1)
                # show(image)
                # plt.show()
                loss, _ = optimization_step(
                    objective_function,
                    image,
                    box_size,
                    noise,
                    crops_per_iteration,
                    model_input_size,
                )
            grads_spec, grads_image = tape.gradient(loss, [spectrum, image])
            # grads_spec = tape.gradient(loss, spectrum)
            return grads_spec, grads_image, image
            # return grads_spec, None, image

        for step in tqdm(range(total_steps)):
            grads_spec, grads_image, image = sstep(spectrum)
            # print('grads spec?', grads_spec.shape)# 'grads img?', grads_image.shape)
            # if step % 200 == 0:
            #  set_size(3, 3)
            #  plt.imshow(image / 2.0 + 0.5)
            #  plt.axis('off')
            #  plt.show()
            optimizer.apply_gradients(zip([grads_spec], [spectrum]))
            transparency_accumulator += tf.abs(grads_image)

    final_image = fourier_preconditioner(spectrum, spectrum_scaler, values_range)
    return final_image, transparency_accumulator


most_important_concept = np.argsort(imp)[::-1][:10]

set_size(10, 10)

for j in range(10):
    v = V[[most_important_concept[j]]][None, :]

    def objective(images):
        a = features(images)
        a = tf.reduce_mean(a, (1, 2))
        y = dot_cossim(a, v)
        return tf.reduce_mean(y)

    image, alpha = fourier(
        objective,
        total_steps=1280,
        image_size=3000,
        learning_rate=0.1,
        decay_power=1.75,
        noise=0.00,
        box_size=(0.10, 0.30),
        values_range=(0.1, 0.9),
    )
    # image, alpha = fourier(objective, total_steps=500, image_size=800,
    #                     learning_rate=1e-3, decay_power=1.8, noise=0.08,
    #                      box_size=(0.04, 0.18))

    plot_maco(image, alpha)
    plt.savefig(
        f"Salicaceae_feature-viz_concept_{most_important_concept[j]}.png",
        dpi=400,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )
    plt.clf()
    plt.close()
    # plt.show()
