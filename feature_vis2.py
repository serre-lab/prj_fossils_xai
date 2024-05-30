import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
import helpers
import matplotlib.pyplot as plt
import xplique
from xplique.features_visualizations import Objective, optimize

# model_path = "./models/resnet_model-8.h5"
model_path = "./models/BEITmodel-9.h5"
# base_model, model = helpers.get_resnet_model('Resnet101')
# model.load_weights(model_path)
cce = tf.keras.losses.categorical_crossentropy
# model = tf.keras.models.load_model(model_path, custom_objects={"cce": cce})
model,_,_ = helpers.get_model(model_path)
# model = tf.keras.models.load_model(model_path)
# import ipdb;ipdb.set_trace()

classes = [
           (0,    'Anacardiaceae - 0'),
           (1,   'Berberidaceae - 1'),
           (2,   'Betulaceae - 2'),
           (3,  'Cupressaceae - 3'),
           (4,    'Dryopteridaceae - 4'),
           (5,   'Fabaceae - 5'),
           (6,   'Fagaceae - 6'),
           (7,  'Juglandaceae - 7'),
           (8,    'Lauraceae - 8'),
           (9,   'Meliaceae - 9'),
           (10,   'Myrtaceae - 10'),
           (11,  'Pinaceae - 11')
           ]

# model.layers[-1].activation = tf.keras.activations.linear

obj_logits = Objective.neuron(model, -1, [c_id for c_id, c_name in classes])
imgs, _ = optimize(obj_logits,
                   nb_steps=2048, # number of iterations
                   optimizer=tf.keras.optimizers.Adam(0.05))


plt.rcParams["figure.figsize"] = [12, 8]
for i in range(len(classes)):
  plt.subplot(len(classes)//4, 4, i+1)
  plt.imshow(imgs[0][i])
  plt.title(classes[i][1])
  plt.axis('off')
  plt.savefig('feature_vis_beit2.png')