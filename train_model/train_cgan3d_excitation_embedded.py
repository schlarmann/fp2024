import keras
import time
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
epochs = 500 #10
antenna_labels = 3
antenna_wires = 60
data_points_per_wire = 2
data_per_point = 3
latent_dim = 64
data_file = "normalized_output.csv"

learning_rate_d    = 0.00025
learning_rate_g    = 0.000005
learning_rate = 0.0002
beta1 = 0.9
beta2 = 0.999

momentum = 0.9 
alpha = 0.2
dropout = 0.4

def renorm(val):
    return (val+1)/2

def load_real_samples():
    global antenna_labels, antenna_wires, data_points_per_wire
    antenna_train,label_train = [],[]
    excitation_train = []
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
        label = [indata[0], indata[1], indata[2]] # remove the scores
        label_train.append(label)
        excitation_train.append(indata[6])
    antenna_train = np.array(antenna_train).astype('float32').reshape(-1,antenna_wires,data_points_per_wire,data_per_point,1)
    return [antenna_train,np.array(label_train),np.array(excitation_train)]

dataset = load_real_samples()
print(dataset[0].shape, dataset[1].shape, dataset[2].shape)

# Create the discriminator.
def build_discriminator():
    global antenna_labels, antenna_wires, data_points_per_wire, data_per_point, momentum, alpha, dropout
    stride_transpose = (2,1,1)
    stride = (2,1,1)
    stride2 = (3,1,3)
    kernel_size = (4,2,3)
    in_shape = (antenna_wires, data_points_per_wire, data_per_point, 1)

    # label input
    in_label = keras.layers.Input(shape=(antenna_labels,), name="Label Input")
    n_nodes = 20*2*1
    label = layers.Dense(n_nodes)(in_label)
    label = layers.Reshape((20,2,1, 1))(label)

    in_image = keras.layers.Input(shape=in_shape, name="Antenna Input")
    n_nodes_image = antenna_wires*data_points_per_wire*data_per_point

    excitation = keras.layers.Input(shape=(1,), name="Excitation Input")
    exc = layers.Embedding(antenna_wires, antenna_wires)(label)
    exc = layers.Dense(n_nodes_image)(exc)
    exc = layers.Reshape((antenna_wires,data_points_per_wire,data_per_point, 1))(exc)
    merge = layers.Concatenate()([in_image, exc])
    
    #disc = layers.Conv3D(64, kernel_size, strides=stride, padding='same')(in_image)
    #disc = layers.LeakyReLU(alpha=alpha)(disc)
    #disc = layers.BatchNormalization(momentum=momentum)(disc)

    #disc = layers.Conv3D(128, kernel_size, strides=stride2, padding='same')(merge)
    #disc = layers.LeakyReLU(alpha=alpha)(disc)
    #disc = layers.BatchNormalization(momentum=momentum)(disc)

    disc = layers.Conv3D(32, kernel_size, strides=stride, padding='same')(disc)
    disc = layers.LeakyReLU(alpha=alpha)(disc)
    disc = layers.BatchNormalization(momentum=momentum)(disc)

    merge = layers.Concatenate()([disc, label])
    disc = layers.Conv3D(24, kernel_size, padding='same')(merge)
    disc = layers.LeakyReLU(alpha=alpha)(disc)
    disc = layers.BatchNormalization(momentum=momentum)(disc)

    disc = layers.Flatten()(disc)
    disc = layers.Dense(32)(disc)
    disc = layers.LeakyReLU(alpha=alpha)(disc)
    disc = layers.Dense(24)(disc)
    disc = layers.LeakyReLU(alpha=alpha)(disc)

    #disc = layers.Dropout(0.4)(disc)

    out_layer = layers.Dense(1, activation='sigmoid')(disc)

    model = keras.models.Model([in_image, in_label], out_layer)
    model.name = "discriminator"
    #opt = keras.optimizers.Adam(learning_rate=0.0003)
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

discriminator = build_discriminator()
discriminator.summary()
#print(discriminator.trainable_weights)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate_d, beta_1=beta1, beta_2=beta2)

# Create the generator.
def build_generator():
    global latent_dim, antenna_labels, antenna_wires, data_points_per_wire, data_per_point, momentum, alpha

    stride_transpose1 = (3,1,1)
    stride_transpose2 = (1,data_points_per_wire,1)
    stride_transpose = (2,1,1)
    kernel_size = (6,2,3)
    kernel_size1 = (4,2,2)
    # load labels
    in_label = keras.layers.Input(shape=(antenna_labels,), name="Label Input")
    label = layers.Dense(latent_dim)(in_label)
    in_latent = keras.layers.Input(shape=(latent_dim,), name="Latent Input")
    latent = layers.Dense(latent_dim)(in_label)

    merge = layers.Concatenate()([latent, label])

    excitation = keras.layers.Input(shape=(1,), name="Excitation Input")
    exc = layers.Embedding(antenna_wires, antenna_wires)(label)
    n_nodes_image = antenna_wires*data_points_per_wire*data_per_point
    exc = layers.Dense(n_nodes_image)(exc)
    exc = layers.Reshape((antenna_wires,data_points_per_wire,data_per_point, 1))(exc)

    n_nodes = latent_dim * 5 * data_points_per_wire * data_per_point

    #expanding random noise vector and converting in 7x7x128 image
    gen = layers.Dense(n_nodes)(merge)
    #gen = layers.LeakyReLU(alpha=alpha)(gen)
    gen = layers.Reshape((5, data_points_per_wire, data_per_point, latent_dim))(gen)

    #Creating features in output by upsampling 
    gen = layers.Conv3DTranspose(512, kernel_size, strides=stride_transpose1, padding='same', use_bias=False)(gen)
    #gen = layers.Conv3D(512, kernel_size1, padding='same', use_bias=False)(gen)
    #gen = layers.LeakyReLU(alpha=alpha)(gen)
    gen = layers.ReLU()(gen)
    #gen = layers.ELU()(gen)
    gen = layers.BatchNormalization(momentum=momentum)(gen)

    #gen = layers.Conv3DTranspose(256, kernel_size, strides=stride_transpose2, padding='same', use_bias=False)(gen)
    gen = layers.Conv3D(256, kernel_size, padding='same', use_bias=False)(gen)
    gen = layers.ReLU()(gen)
    gen = layers.BatchNormalization(momentum=momentum)(gen)

    gen = layers.Conv3DTranspose(128, kernel_size, strides=stride_transpose, padding='same', use_bias=False)(gen)
    gen = layers.ReLU()(gen)
    gen = layers.BatchNormalization(momentum=momentum)(gen)
    
    gen = layers.Conv3DTranspose(64, kernel_size, strides=stride_transpose, padding='same', use_bias=False)(gen)
    gen = layers.ReLU()(gen)
    gen = layers.BatchNormalization(momentum=momentum)(gen)
    
    #gen = layers.Conv3D(128, kernel_size, padding='same', use_bias=False)(gen)
    #gen = layers.LeakyReLU(alpha=alpha)(gen)
    #gen = layers.BatchNormalization(momentum=momentum)(gen)
    
    #gen = layers.Conv3D(64, kernel_size1, padding='same', use_bias=False)(gen)
    #gen = layers.LeakyReLU(alpha=alpha)(gen)
    #gen = layers.BatchNormalization(momentum=momentum)(gen)

    merge_ex = layers.Concatenate()([gen, exc])

    gen = layers.Conv3D(32, kernel_size, padding='same', use_bias=False)(merge_ex)
    gen = layers.ReLU()(gen)
    #gen = layers.BatchNormalization(momentum=momentum)(gen)
    
    gen = layers.Conv3D(16, kernel_size, padding='same', use_bias=False)(gen)
    gen = layers.ReLU()(gen)
    #gen = layers.BatchNormalization(momentum=momentum)(gen)

    #Converting final output to a 1 channel output
    out = layers.Conv3D(1, kernel_size, padding='same', use_bias=False)(gen)
    out = layers.Activation("tanh")(out)
    #out_layer = layers.AveragePooling2D((8,8))(conv)
    model = keras.models.Model([in_latent, in_label], out)
    model.name = "generator"
    #opt = keras.optimizers.Adam(learning_rate=0.0003)
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

generator = build_generator()
generator.summary()
generator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate_g, beta_1=beta1, beta_2=beta2)
#print(generator.trainable_weights)

loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = loss_function(tf.ones_like(real_output), real_output)
    # real_loss will quantify our loss to distinguish the real images
    
    fake_loss = loss_function(tf.zeros_like(fake_output), fake_output)
    # fake_loss will quantify our loss to distinguish the fake images (generated)
    
    # Real image = 1, Fake image = 0 (array of ones and zeros)
    total_loss = real_loss + fake_loss
    return total_loss
def generator_loss(fake_output):
    # We want the false images to be seen as real images (1)
    return loss_function(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be converted 
# from Eager mode of Tensorflow (easier to code but slower to execute) 
# to Graph mode (harder to code but faster to execute)

@tf.function
def train_step(images, labels, excitations):
    noise = tf.random.normal([batch_size, latent_dim])

    # To make sure we know what is done, we will use a gradient tape instead of compiling
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Training the generator
        generated_images = generator([noise, labels, excitations] , training=True) 

        # Training the discriminator
        real_output = discriminator([images, labels, excitations], training=True)           # Training the discriminator on real images
        fake_output = discriminator([generated_images, labels, excitations], training=True) # Training the discriminator on fake images

        # Calculating the losses
        gen_loss =  generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # Building the gradients
        gradients_of_generator =     gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        # Applying the gradients (backpropagation)
        generator_optimizer.apply_gradients(    zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

g_losses = []
d_losses = []
def train(train_images, train_labels, train_excitations, epochs):

    num_batches = int(train_images.shape[0]/batch_size) # Amount of batches
    for epoch in range(epochs):
        start = time.time() # Timing the epoch

        for batch_idx in range(num_batches): # For each batch
            images = train_images[batch_idx*batch_size : (batch_idx+1)*batch_size]
            labels = train_labels[batch_idx*batch_size : (batch_idx+1)*batch_size]
            excitations = train_excitations[batch_idx*batch_size : (batch_idx+1)*batch_size]
            gen_loss, disc_loss = train_step(images, labels, excitations)
            
            # Saving the losses
            g_losses.append(np.array(gen_loss))  
            d_losses.append(np.array(disc_loss))

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print("Generator loss for last batch: ",g_losses[-1])
        print("Discriminator loss for last batch: ",d_losses[-1])

antenna, labels, excitations = dataset

# Training
try:
    train(antenna, labels, excitations, epochs)
except Exception as e:
    raise(e)
except KeyboardInterrupt as e:
    pass # Save even when we quit

with open(f"log_{time.time()}.csv","w") as f:
    f.write("Epoch, Generator Loss, Discriminator Loss\n")
    for i in range(len(g_losses)):
        f.write(f"{i+1},{g_losses[i]},{d_losses[i]}"+"\n")

# Save the model.
keras.saving.save_model(generator, "generator3d.keras")
keras.saving.save_model(discriminator, "discriminator3d.keras")
keras.saving.save_model(generator, f"generator3d_{time.time()}.keras")
keras.saving.save_model(discriminator, f"discriminator3d_{time.time()}.keras")
