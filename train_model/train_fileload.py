import keras
import time

from keras import layers
from keras import ops
from tqdm import tqdm
import tensorflow as tf
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
DO_TRAIN = True

# Constants / Hyperparameters
batch_size = 96 #96
epochs = 17 #10
antenna_labels = 7
antenna_wires = 60
data_points_per_wire = 2
data_per_point = 3
latent_dim = 128
data_file = "normalized_output.csv"

def load_real_samples():
    global antenna_labels, antenna_wires, data_points_per_wire
    antenna_train,label_train = [],[]
    with open(data_file) as f:
        data = f.readlines()
    for line in data[1:]:
        indata = [float(x.strip()) for x in line.split(",")[:-1]]
        antenna_points = []
        for i in range(60):
            baseidx = antenna_labels + i*8
            antenna_points.append(indata[baseidx+0])
            antenna_points.append(indata[baseidx+1])
            antenna_points.append(indata[baseidx+3])
            antenna_points.append(indata[baseidx+4])
            antenna_points.append(indata[baseidx+5])
            antenna_points.append(indata[baseidx+6])
        antenna_train.append(antenna_points)
        label_train.append(indata[:antenna_labels])
    antenna_train = np.array(antenna_train).astype('float32').reshape(-1,antenna_wires,data_points_per_wire,data_per_point,1)
    return [antenna_train,np.array(label_train)]

dataset = load_real_samples()
print(dataset[0].shape, dataset[1].shape)



discriminator = keras.saving.load_model("discriminator3d.keras")
discriminator.summary()
#print(discriminator.trainable_weights)

generator = keras.saving.load_model("generator3d.keras")
generator.summary()
#print(generator.trainable_weights)

def generate_real_samples(n_samples):
    global dataset
    antenna, labels = dataset
    
    #generating n random samples
    ix = np.random.randint(0, antenna.shape[0], n_samples)
    X, labels = antenna[ix], labels[ix]
    
    #Observe how class-labels alongside binary label(1) is return. 
    y = np.ones((n_samples, 1))
    return [X, labels], y

def generate_latent_vector(n_samples):
    global latent_dim, antenna_labels, antenna_wires, data_points_per_wire
    x_input = np.random.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = np.random.randn(antenna_labels * n_samples)
    labels = labels.reshape(n_samples, antenna_labels)
    return [z_input, labels]

def generate_fake_samples(n_samples):
    global generator, latent_dim
    z_input, labels_input = generate_latent_vector(n_samples)
    images = generator.predict([z_input, labels_input])
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y

def build_cgan(g_model, d_model):
    d_model.trainable = False
    gen_noise, gen_label = g_model.input
    gen_output = g_model.output
    gan_output = d_model([gen_output, gen_label])
    model = keras.models.Model([gen_noise, gen_label], gan_output)
    model.name = "CGAN"
    opt = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

cgan = build_cgan(generator, discriminator)
cgan.summary()
#print(cgan.trainable_weights)

csvLines = []

if DO_TRAIN:
    # Training
    batch_per_epo = int( 1 * (dataset[0].shape[0] / batch_size) )
    half_batch = int(batch_size / 2)
    for i in range(epochs):
        for j in tqdm(range(batch_per_epo)):
            #generate real sample
            [X_real, labels_real], y_real = generate_real_samples(half_batch)
            #train discriminator on real dataset
            d_loss1, _ = discriminator.train_on_batch([X_real, labels_real], y_real)
            #generate fake sample
            [X_fake, labels], y_fake = generate_fake_samples(half_batch)
            #train discriminator on fake dataset
            d_loss2, _ = discriminator.train_on_batch([X_fake, labels], y_fake)
            #Training CGAN
            [z_input, labels_input] = generate_latent_vector(batch_size)
            y_gan = np.ones((batch_size, 1))
            g_loss = cgan.train_on_batch([z_input, labels_input], y_gan)
        csvStr = f"{i},{d_loss1},{d_loss2},{g_loss[0]},{g_loss[1]},{g_loss[2]},"
        csvLines.append(csvStr)
        print('>Epoch, Loss Disc.1, Loss Disc.2, Loss Gen : '+csvStr)

with open(f"log_{time.time()}.csv","w") as f:
    f.write("Epoch, Loss Disc.1, Loss Disc.2, Loss Gen 1, Loss Gen 2, Loss Gen 3\n")
    for line in csvLines:
        f.write(line+"\n")

# Save the model.
keras.saving.save_model(generator, "generator3d.keras")
keras.saving.save_model(discriminator, "discriminator3d.keras")
keras.saving.save_model(generator, f"generator3d_{time.time()}.keras")
keras.saving.save_model(discriminator, f"discriminator3d_{time.time()}.keras")
