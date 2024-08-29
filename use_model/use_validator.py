import keras
import necFile
import necLabeler
import random

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
data_file = "normalized_validation_output.csv"

learning_rate_d    = 0.00025
learning_rate_g    = 0.000005 # 0.00001
learning_rate = 0.0002
beta1 = 0.9
beta2 = 0.999

momentum = 0.9 
alpha = 0.2
dropout = 0.4

discriminator = keras.models.load_model("model/discriminator3d.keras")
discriminator.summary()
#print(generator.trainable_weights)

def renorm(val):
    return (val+1)/2
def load_real_samples():
    global antenna_labels, antenna_wires, data_points_per_wire
    antenna_train,label_train = [],[]
    with open(data_file) as f:
        data = f.readlines()
    used = data[1:]
    random.shuffle(used)
    for line in used:
        indata = [float(x.strip()) for x in line.split(",")[:-1]]
        antenna_points = []
        for i in range(60):
            baseidx = antenna_labels + i*8
            antenna_points.append(renorm(indata[baseidx+0]))
            antenna_points.append(renorm(indata[baseidx+1]))
            antenna_points.append(renorm(indata[baseidx+3]))
            antenna_points.append(renorm(indata[baseidx+4]))
            antenna_points.append(renorm(indata[baseidx+5]))
            antenna_points.append(renorm(indata[baseidx+6]))
        antenna_train.append(antenna_points)
        # maxGain, frontBackRatio, absImpedance, freqScore, antennaLengthScore, greatestExtentScore, excitationSegment
        label = [indata[0], indata[1], indata[2], indata[6]] # remove the scores
        label_train.append(label)
    antenna_train = np.array(antenna_train).astype('float32').reshape(-1,antenna_wires,data_points_per_wire,data_per_point,1)
    return [antenna_train,np.array(label_train)]

dataset = load_real_samples()
print(dataset[0].shape, dataset[1].shape)


MAX_GAIN = 100
MAX_FBR = 40
MAX_IMPEDANCE = 10000 # Ohms
MAX_WIRES = 60

nl = necLabeler.necLabeler()


sumRec = 0
for i in range(len(dataset[0])):
    in_antenna = np.array([dataset[0][i]])
    in_label = np.array([dataset[1][i]])
    print(f"Input labels: Max Gain: {dataset[1][i][0]*MAX_GAIN:0.3f}, Front/Back Ratio: {dataset[1][i][1]*MAX_FBR:0.3f}, Impedance: {dataset[1][i][2]*MAX_IMPEDANCE:0.3f}, Excitation: {int(dataset[1][i][3]*MAX_WIRES)}")
    X = discriminator.predict([in_antenna, in_label])
    print(X)
    sumRec += round(X[0][0])

print(f"Recognized: {sumRec}/{len(dataset[0])}")

