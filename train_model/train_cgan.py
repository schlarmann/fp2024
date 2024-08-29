import keras

from keras import layers
from keras import ops
from tqdm import tqdm
import tensorflow as tf
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
DO_TRAIN = True

# Constants / Hyperparameters
batch_size = 128 #96
epochs = 15 #10
antenna_labels = 7
antenna_wires = 60
data_points_per_wire = 2*3+2
latent_dim = 128
data_file = "normalized_output.csv"

def load_real_samples():
    global antenna_labels, antenna_wires, data_points_per_wire
    antenna_train,label_train = [],[]
    with open(data_file) as f:
        data = f.readlines()
    for line in data[1:]:
        indata = [float(x.strip()) for x in line.split(",")[:-1]]
        antenna_train.append(indata[antenna_labels:])
        label_train.append(indata[:antenna_labels])
    antenna_train = np.array(antenna_train).astype('float32').reshape(-1,antenna_wires,data_points_per_wire,1)
    return [antenna_train,np.array(label_train)]

dataset = load_real_samples()
print(dataset[0].shape, dataset[1].shape)

# Create the discriminator.
def build_discriminator():
    global antenna_labels, antenna_wires, data_points_per_wire
    in_shape = (antenna_wires, data_points_per_wire, 1)

    # label input
    in_label = keras.layers.Input(shape=(antenna_labels,))
    n_nodes = in_shape[0] * in_shape[1]
    label = layers.Dense(n_nodes)(in_label)
    label = layers.Reshape((in_shape[0], in_shape[1], 1))(label)

    in_image = keras.layers.Input(shape=in_shape)
    merge = layers.Concatenate()([in_image, label])
    
    disc = layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
    disc = layers.LeakyReLU(alpha=0.2)(disc)
    disc = layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(disc)
    disc = layers.LeakyReLU(alpha=0.2)(disc)
    disc = layers.Flatten()(disc)
    disc = layers.Dropout(0.4)(disc)
    out_layer = layers.Dense(1, activation='sigmoid')(disc)
    model = keras.models.Model([in_image, in_label], out_layer)
    model.name = "discriminator"
    opt = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

discriminator = build_discriminator()
discriminator.summary()
#print(discriminator.trainable_weights)

# Create the generator.
def build_generator():
    global latent_dim, antenna_labels, antenna_wires, data_points_per_wire
    # load labels
    in_label = keras.layers.Input(shape=(antenna_labels,))
    n_nodes = antenna_wires * data_points_per_wire

    #Expanding class-label embedding
    label = layers.Dense(n_nodes)(in_label)

    #Converting flat array as 2d image 
    label = layers.Reshape((antenna_wires, data_points_per_wire, 1))(label)

    in_latent = keras.layers.Input(shape=(latent_dim,))
    n_nodes = latent_dim * antenna_wires * data_points_per_wire

    #expanding random noise vector and converting in 7x7x128 image
    gen = layers.Dense(n_nodes)(in_latent)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Reshape((antenna_wires, data_points_per_wire, latent_dim))(gen)

    #Adding class-label 2d image to this random noise image
    merge = layers.Concatenate()([gen, label])

    #Creating features in output by upsampling 
    gen = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)

    #Converting final output to a 1 channel output
    conv = layers.Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)
    out_layer = layers.AveragePooling2D((8,8))(conv)
    model = keras.models.Model([in_latent, in_label], out_layer)
    model.name = "generator"
    opt = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

generator = build_generator()
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
        print('>Loss Discriminator: {}, {} , Generator: {}'.format(d_loss1,d_loss2,g_loss))
 

# Save the model.
keras.saving.save_model(generator, "generator.keras")
keras.saving.save_model(discriminator, "discriminator.keras")