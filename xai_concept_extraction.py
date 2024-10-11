import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from math import ceil
from sklearn.decomposition import NMF
import cv2
import xplique
from xplique.features_visualizations import Objective
from xplique.features_visualizations import maco
#from xplique.plot import plot_maco
from tqdm import tqdm

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

fossils_dir = '/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Florissant_Fossil_v2.0'
leaves_dir = '/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Extant_Leaves'
plot_save_dir = '/cifs/data/tserre_lrs/projects/prj_fossils_xai/concepts/fossils_concept'
plot_leaves_save_dir = '/cifs/data/tserre_lrs/projects/prj_fossils_xai/concepts/leaves_concept'
save_feature_viz = '/cifs/data/tserre_lrs/projects/prj_fossils_xai/maco/fossils_viz'
save_leaves_feature_viz = '/cifs/data/tserre_lrs/projects/prj_fossils_xai/maco/leaves_viz'
mask_dir = '/cifs/data/tserre_lrs/projects/prj_fossils_xai/mask_images_all_leaves2'
model_path = '/cifs/data/tserre_lrs/projects/prj_fossils_xai/new_models/model-14_RESNET_101_TRIPLET.h5'

classes = ['Anacardiaceae', 'Berberidaceae', 'Betulaceae', 'Cupressaceae', 'Dryopteridaceae', 'Fabaceae', 'Fagaceae', 'Juglandaceae', 'Lauraceae', 'Meliaceae', 'Myrtaceae', 'Pinaceae', 'Rhamnaceae', 'Rosaceae', 'Salicaceae', 'Sapindaceae']

cce = tf.keras.losses.categorical_crossentropy
model = tf.keras.models.load_model(model_path, custom_objects= {'cce': cce})
print(model.summary())
features = tf.keras.Model(model.input, model.layers[-5].output)

def load_fossils_dir(class_name, fossils_dir):
	class_dir = os.path.join(fossils_dir, class_name)
	paths = os.listdir(class_dir)
	fossils = []
	count = 0
	for p in paths:
		fossils_path = os.path.join(class_dir, p)
		img = cv2.imread(fossils_path)[...,::-1]
		img = img.astype(np.float32)
		fossils.append(img)
		count+=1
	return fossils

def load_leaves_dir(class_name, leaves_dir, mask_dir):
  mask_dir = os.path.join(mask_dir, class_name)
  leaves_dir = os.path.join(leaves_dir, class_name)
  paths = os.listdir(mask_dir)
  masked_imgs = []
  imgs = []
  count = 0
  for p in paths[:100]:
    mask_path = os.path.join(mask_dir, p)
    mask = cv2.imread(mask_path)/225.0
    img = cv2.imread(image_dir + "/" + p)[...,::-1]
    img = img.astype(np.float32)
    image = img * (mask > 0.1).astype(np.float32)
    masked_imgs.append(image)
    imgs.append(img)
    count+=1
  return imgs, masked_imgs
	

def generate_square_crops(image, crop_size=1000):
    height, width, _ = image.shape
    crops = []
    y_steps = ceil(height / crop_size)
    x_steps = ceil(width / crop_size)
    threshold = 0.9
    for y in range(y_steps):
        for x in range(x_steps):
            start_y = y * crop_size
            end_y = min(start_y + crop_size, height)
            start_x = x * crop_size
            end_x = min(start_x + crop_size, width)

            # If we are at the end, take more from the other side
            if end_y - start_y < crop_size:
                start_y = max(0, end_y - crop_size)
            if end_x - start_x < crop_size:
                start_x = max(0, end_x - crop_size)

            crop = image[start_y:end_y, start_x:end_x, :]
            crops.append(crop)


    return np.array(crops)

def preprocess(x):
	return x/255.0
	
def get_importance(U, V, class_id):
  return (W[:, class_id] @ V.T) * np.mean(U, 0)
  
def set_size(w,h):
  plt.rcParams["figure.figsize"] = [w,h]
  
def show(img, **kwargs):
  img  = np.array(img).astype(np.float32)
  img -= img.min()
  img /= img.max()
  plt.imshow(img, **kwargs)
  plt.axis('off')

def plot_concept(u, crops, id):
  best_idx = np.argsort(u[:, id])[::-1][:10]

  for i in range(10):
    plt.subplot(2, 5, i+1)
    c = crops[best_idx[i]]
    show(c)

    h = U_big[best_idx[i]][:, :, id]
    h = cv2.resize(h, (c.shape[0], c.shape[1]), interpolation=cv2.INTER_CUBIC)
    h = h / h.max()
    show(h, cmap='jet', alpha=0.25)

def save_concept(u, crops, id, concept_dir):
  best_idx = np.argsort(u[:, id])[::-1][:10]
  plt.tight_layout()
  for i in range(10):
    c = crops[best_idx[i]]
    show(c)

    h = U_big[best_idx[i]][:, :, id]
    h = cv2.resize(h, (c.shape[0], c.shape[1]), interpolation=cv2.INTER_CUBIC)
    h = h / h.max()
    show(h, cmap='jet', alpha=0.25)
    plt.savefig(f'{concept_dir}/{i}.png')
    


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

    #image = image * alpha

    # overlay alpha mask on the image
    plt.imshow(np.concatenate([image, alpha], -1))
    plt.axis('off')
    #plt.show()

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
    image /= (image.std()+1e-3)
    return image

def clip_percentile(img, p=0.1):
    # clip pixel values to specified percentile range
    return np.clip(img, np.percentile(img, p), np.percentile(img, 100-p))

def show(img, **kwargs):
    # display image with normalization and channels in the last dimension
    img = check_format(img)
    img = normalize(img)

    plt.imshow(img, **kwargs)
    plt.axis('off')
    #plt.show()
      
def cosine_similarity(tensor_a, tensor_b):
    # Calculate cosine similarity
    norm_dims = list(range(1, len(tensor_a.shape)))
    tensor_a = tf.math.l2_normalize(tensor_a, axis=norm_dims)
    tensor_b = tf.math.l2_normalize(tensor_b, axis=norm_dims)
    return tf.reduce_sum(tensor_a * tensor_b, axis=norm_dims)

def dot_cossim(tensor_a, tensor_b, cossim_pow=2.0):
    # Compute dot product scaled by cosine similarity
    cosim = tf.math.pow(tf.clip_by_value(cosine_similarity(tensor_a, tensor_b), 1e-1, 1.0), cossim_pow)
    dot = tf.reduce_sum(tensor_a * tensor_b)
    return dot * cosim
    
# tensor for color correlation svd square root
color_correlation_svd_sqrt = tf.constant(
    [[0.56282854, 0.58447580, 0.58447580],
     [0.19482528, 0.00000000, -0.19482528],
     [0.04329450, -0.10823626, 0.06494176]],
    dtype=tf.float32
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
def optimization_step(objective_function, image, box_size, noise_level, number_of_crops_per_iteration, model_input_size):
    # performs an optimization step on the generated image
    assert box_size[1] >= box_size[0]
    assert len(image.shape) == 3
    assert image.shape[-1] == 3


    # generate random boxes
    x0 = 0.5 + tf.random.normal((number_of_crops_per_iteration,)) * 0.15
    y0 = 0.5 + tf.random.normal((number_of_crops_per_iteration,)) * 0.15
    delta_x = tf.random.uniform((number_of_crops_per_iteration,)) * (box_size[1] - box_size[0]) + box_size[0]
    delta_y = delta_x

    box_indices = tf.zeros(shape=(number_of_crops_per_iteration,), dtype=tf.int32)
    boxes = tf.stack([x0 - delta_x * 0.5,
                      y0 - delta_y * 0.5,
                      x0 + delta_x * 0.5,
                      y0 + delta_y * 0.5], -1)

    crops = tf.image.crop_and_resize(image[None, :, :, :], boxes, box_indices,
                                      (model_input_size, model_input_size))

    score = objective_function(crops)
    loss = -score

    return loss, image
 

def fft_2d_freq(width: int, height: int) -> np.ndarray:
    freq_y = np.fft.fftfreq(height)[:, np.newaxis].astype(np.float64)

    cut_off = int(width % 2 == 1)
    freq_x = np.fft.fftfreq(width)[:width//2+1+cut_off]

    return np.sqrt(freq_x**2 + freq_y**2)

def get_fft_scale(width: int, height: int, decay_power: float = 1.0) -> tf.Tensor:
    frequencies = fft_2d_freq(width, height)
    fft_scale = 1.0 / np.maximum(frequencies, 1.0 / max(width, height)) ** decay_power
    fft_scale = fft_scale * np.sqrt(width * height)

    return tf.cast(fft_scale, dtype=tf.complex64)

def init_olah_buffer(width, height, std=1e-3):
    # Initialize the Olah buffer with a random spectrum
    spectrum_shape = (3, width, height // 2 + 1)
    random_spectrum = tf.complex(tf.random.normal(spectrum_shape) * std, tf.random.normal(spectrum_shape) * std)
    return random_spectrum

def fourier_preconditioner(spectrum, spectrum_scaler, values_range):
    # Precondition the Fourier spectrum and convert it to spatial domain
    assert spectrum.shape[0] == 3

    #spectrum = standardize_complex(spectrum)
    spectrum = spectrum * spectrum_scaler

    spatial_image = tf.signal.irfft2d(spectrum)
    spatial_image = tf.transpose(spatial_image, [1,2,0])

    image = spatial_image
    image = standardize(image) / 2.0
    image = recorrelate_colors(image)
    #color_recorrelated_image = spatial_image
    #image = spatial_image
    #image = color_recorrelated_image

    image = tf.nn.sigmoid(image)
    #image = image - tf.reduce_min(image)
    #image = image / (tf.reduce_max(image) + 1e-3)
    image = image * (values_range[1] - values_range[0]) + values_range[0]

    #image = tf.sigmoid(image) * (values_range[1] - values_range[0]) + values_range[0]

    #mean = tf.reduce_mean(image, (0, 1))
    #image = image - (image - mean[None, None, :]) * 0.5
    #image = image * (values_range[1] - values_range[0]) + values_range[0]
    return image


def fourier(objective_function, decay_power=1.5, total_steps=1000, learning_rate=1.0, image_size=1280, model_input_size=384,
            noise=0.08, values_range=(-0.1, 1.1), crops_per_iteration=8, box_size=(0.15, 0.25), device='/GPU:0'):
    # Perform the Olah optimization process
    assert values_range[1] >= values_range[0]
    assert box_size[1] >= box_size[0]

    spectrum = init_olah_buffer(image_size, image_size, std=1.0)
    spectrum_scaler = get_fft_scale(image_size, image_size, decay_power)

    with tf.device(device):
        spectrum = tf.Variable(spectrum)
        optimizer = tf.optimizers.Nadam(learning_rate=learning_rate)

        transparency_accumulator = tf.zeros((image_size, image_size, 3), dtype=tf.float32)

        @tf.function
        def sstep(spectrum):
          with tf.GradientTape() as tape:
            tape.watch(spectrum)
            image = fourier_preconditioner(spectrum, spectrum_scaler, values_range)
            tape.watch(image)
            #set_size(1, 1)
            #show(image)
            #plt.show()
            loss, _ = optimization_step(objective_function, image, box_size, noise, crops_per_iteration, model_input_size)
          grads_spec, grads_image = tape.gradient(loss, [spectrum, image])
          #grads_spec = tape.gradient(loss, spectrum)
          return grads_spec, grads_image, image
          #return grads_spec, None, image

        for step in tqdm(range(total_steps)):
            grads_spec, grads_image, image = sstep(spectrum)
            #print('grads spec?', grads_spec.shape)# 'grads img?', grads_image.shape)
            #if step % 200 == 0:
            #  set_size(3, 3)
            #  plt.imshow(image / 2.0 + 0.5)
            #  plt.axis('off')
            #  plt.show()
            optimizer.apply_gradients(zip([grads_spec], [spectrum]))
            transparency_accumulator += tf.abs(grads_image)

    final_image = fourier_preconditioner(spectrum, spectrum_scaler, values_range)
    return final_image, transparency_accumulator
    

for i in range(len(classes)):
	class_id = i
	nb_concepts = 40 
	CROPS = []
	ACTIVATIONS = []

	class_save_dir_c = os.path.join(plot_save_dir, classes[class_id], 'coalesce')
	class_save_dir_ind = os.path.join(plot_save_dir, classes[class_id], 'individual')
	class_viz_dir = os.path.join(save_feature_viz, classes[class_id])
	os.makedirs(class_save_dir_c, exist_ok = True)
	os.makedirs(class_save_dir_ind, exist_ok = True)
	os.makedirs(class_viz_dir, exist_ok = True)

	print(f'Fossils Dir: {fossils_dir}')
	print(f'Plot Dir: {plot_save_dir}')
	print(f'Class Save Dir: {class_save_dir_c}')
	print(f'class save dir ind: {class_save_dir_ind}')
	print(f'ViZ dir: {class_viz_dir}')

	X = load_fossils_dir(classes[class_id], fossils_dir)

	count = 0
    for i,x in enumerate(X):
        crops = generate_square_crops(x)
        crops = tf.image.resize(crops, (384, 384))
        CROPS += list(crops.numpy().astype(np.uint8))
        crops = preprocess(crops)
        activations = features(crops)
        ACTIVATIONS += list(activations.numpy())
        count+=1

    ACTIVATIONS = np.array(ACTIVATIONS)
    print(ACTIVATIONS.shape)

    nmf = NMF(n_components = nb_concepts, init = 'random', random_state = 0)

    A = np.array(np.mean(ACTIVATIONS, (1,2)))
    U = nmf.fit_transform(A)
    V = nmf.components_
    W = np.array(model.layers[-1].weights[0])

    imp = get_importance(U,V, class_id)

    U_big = nmf.transform(ACTIVATIONS.reshape((-1, 2048)))
    U_big = U_big.reshape((-1, 12, 12, nb_concepts))
    #set_size(7, 4)
    most_important_concept = np.argsort(imp)[::-1][:10]

    import ipdb;ipdb.set_trace()
    for mic in most_important_concept:
        importance_val = imp[mic]/np.sum(imp)
        concept_dir = os.path.join(class_save_dir_ind, f'{mic}_{importance_val}')
        os.makedirs(concept_dir, exist_ok = True)
        save_concept(U, CROPS, mic, concept_dir)

    set_size(10, 10)
    for j in range(10):
        v = V[[most_important_concept[j]]][None, :]
        def objective(images):
            a = features(images)
            a = tf.reduce_mean(a, (1,2))
            y = dot_cossim(a, v)
            return tf.reduce_mean(y)
        image, alpha = fourier(objective, total_steps=1280, image_size=3000,learning_rate=0.1, decay_power=1.75, noise=0.00,box_size=(0.10, 0.30), values_range=(0.1, 0.9))
        plot_maco(image, alpha)
        plt.savefig(f'{class_viz_dir}/{classes[class_id]}_concept_{most_important_concept[j]}.png', dpi=400, bbox_inches='tight',transparent=True, pad_inches=0)
        plt.clf()
        plt.close()

