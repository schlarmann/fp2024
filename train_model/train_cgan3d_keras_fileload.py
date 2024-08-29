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


discriminator = keras.saving.load_model("discriminator3d.keras")
discriminator.summary()

discriminator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate_d, beta_1=beta1, beta_2=beta2)

generator = keras.saving.load_model("generator3d.keras")
generator.summary()

generator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate_g, beta_1=beta1, beta_2=beta2)

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

# Notice the use of `tf.function`
# This annotation causes the function to be converted 
# from Eager mode of Tensorflow (easier to code but slower to execute) 
# to Graph mode (harder to code but faster to execute)

@tf.function
def train_step(images, labels):
    noise = tf.random.normal([batch_size, latent_dim])

    # To make sure we know what is done, we will use a gradient tape instead of compiling
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Training the generator
        generated_images = generator([noise, labels] , training=True) 

        # Training the discriminator
        real_output = discriminator([images, labels], training=True)           # Training the discriminator on real images
        fake_output = discriminator([generated_images, labels], training=True) # Training the discriminator on fake images

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
def train(train_images, train_labels, epochs):

    num_batches = int(train_images.shape[0]/batch_size) # Amount of batches
    for epoch in range(epochs):
        start = time.time() # Timing the epoch

        for batch_idx in range(num_batches): # For each batch
            images = train_images[batch_idx*batch_size : (batch_idx+1)*batch_size]
            labels = train_labels[batch_idx*batch_size : (batch_idx+1)*batch_size]
            gen_loss, disc_loss = train_step(images, labels)
            
            # Saving the losses
            g_losses.append(np.array(gen_loss))  
            d_losses.append(np.array(disc_loss))

        # Produce images for the GIF as we go
        #display.clear_output(wait=True)
        #generate_and_save_images(generator,
        #                         epoch + 1,
        #                         seed,
        #                         g_losses,
        #                         d_losses,
        #                         conditions=seed_labels,
        #                         x_axis='total')

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print("Generator loss for last batch: ",g_losses[-1])
        print("Discriminator loss for last batch: ",d_losses[-1])

    # Generate after the final epoch
    #display.clear_output(wait=True)
    #generate_and_save_images(generator,
    #                           epochs,
    #                           seed,
    #                           g_losses,
    #                           d_losses,
    #                           conditions=seed_labels,
    #                           x_axis='total')
antenna, labels = dataset
#%%time
# Training
try:
    train(antenna, labels, epochs)
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
