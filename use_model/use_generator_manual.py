import keras
import necFile

from keras import layers
from keras import ops
from tqdm import tqdm
import tensorflow as tf
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
DO_TRAIN = True


# Constants / Hyperparameters
batch_size = 64 #96
epochs = 2500 #10
antenna_labels = 4
antenna_wires = 60
data_points_per_wire = 2
data_per_point = 3
latent_dim = 64
data_file = "normalized_output.csv"

learning_rate_d    = 0.00025
learning_rate_g    = 0.000005 # 0.00001
learning_rate = 0.0002
beta1 = 0.9
beta2 = 0.999

momentum = 0.9 
alpha = 0.2
dropout = 0.4

nf = necFile.necFile()

generator = keras.models.load_model("model/generator3d.keras")
generator.summary()
#print(generator.trainable_weights)

def generate_latent_vector(n_samples):
    global latent_dim, antenna_labels, antenna_wires, data_points_per_wire
    x_input = np.random.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = np.random.randn(antenna_labels * n_samples)
    labels = labels.reshape(n_samples, antenna_labels)
    return [z_input, labels]
maxGain = 20
frontBackRatio = 15
absImpedance = 35
excitation = 30


noise = tf.random.normal([1, latent_dim])
MAX_GAIN = 100
MAX_FBR = 40
MAX_IMPEDANCE = 10000 # Ohms
MAX_WIRES = 60
in_labels = np.array([[maxGain/MAX_GAIN, frontBackRatio/MAX_FBR, absImpedance/MAX_IMPEDANCE, excitation/MAX_WIRES]])
X_fake = generator.predict([noise, in_labels])

#print(X_fake, X_fake.shape)
output = np.reshape(X_fake, (60,6))
#print(output)
fakewires = []
for wire_raw in output:
    wire = []
    for coord in wire_raw:
        wire.append((coord*2)-1)
    wire_start = wire[0:3]
    wire_end = wire[3:6]
    wire_radius = 0.001
    wire_segments = 0.05*3
    realwire = [wire_start, wire_end, wire_radius, wire_segments]
    realwire = wire_start + wire_end + [wire_radius, wire_segments]
    for elem in realwire:
        fakewires.append(elem)

fakewires_np = np.array(fakewires)
params = [maxGain/MAX_GAIN, frontBackRatio/MAX_FBR, absImpedance/MAX_IMPEDANCE, 1, 1, 1, excitation/MAX_WIRES]
vec = params + fakewires

nf.denormalize(vec, 2450, MAX_WIRES)
print(nf)
nf.show()
