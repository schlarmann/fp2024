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
data_file = "normalized_output.csv"

learning_rate_d    = 0.00025
learning_rate_g    = 0.000005 # 0.00001
learning_rate = 0.0002
beta1 = 0.9
beta2 = 0.999

momentum = 0.9 
alpha = 0.2
dropout = 0.4

generator = keras.models.load_model("model/generator3d.keras")
generator.summary()
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

labels = random.sample(dataset[1].tolist(), 10)
print(labels)

MAX_GAIN = 100
MAX_FBR = 40
MAX_IMPEDANCE = 10000 # Ohms
MAX_WIRES = 60

nl = necLabeler.necLabeler()

for in_labels in labels:
    in_label = np.array([in_labels])
    print(f"Input labels: Max Gain: {in_labels[0]*MAX_GAIN:0.3f}, Front/Back Ratio: {in_labels[1]*MAX_FBR:0.3f}, Impedance: {in_labels[2]*MAX_IMPEDANCE:0.3f}, Excitation: {int(in_labels[3]*MAX_WIRES)}")
    noise = tf.random.normal([1, latent_dim])
    X_fake = generator.predict([noise, in_label])

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
    params = [in_labels[0], in_labels[1], in_labels[2], 1, 1, 1, in_labels[3]]
    vec = params + fakewires

    nf = necFile.necFile()
    nf.denormalize(vec, 2450, MAX_WIRES)


    outlabels = []
    for i in range(5): #Retry 3 times
            try:
                nf2 = nl.labelFile(nf)
            except:
                for i in range(len(nf.wires)):
                    nf.wires[i][nf.WIRE_RADIUS] /= 2 # Decrease the radius of the wires
                nl.reset()
            else:
                nl.reset()
                outlabels = nf2.getLabels2()
                break
    print(f"Output labels: Max Gain: {outlabels[0]:0.3f}, Front/Back Ratio: {outlabels[1]:0.3f}, Impedance: {outlabels[2]:0.3f}")
    print(f"{in_labels[0]*MAX_GAIN:0.3f} &  {outlabels[0]:0.3f} & {in_labels[1]*MAX_FBR:0.3f} & {outlabels[1]:0.3f} & {in_labels[2]*MAX_IMPEDANCE:0.3f} & {outlabels[2]:0.3f} \\tabularnewline")
    
    nf.show()

