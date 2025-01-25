# Importing required libraries
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Setting up paths for datasets
INPUT_PATH = "data/input_images/"  # Path to input images
OUTPUT_PATH = "data/output_images/"  # Path to corresponding output images
CHECKPOINT_DIR = "./pix2pix_checkpoints/"  # Directory to save model checkpoints
BATCH_SIZE = 8
IMG_SIZE = (256, 256)

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=IMG_SIZE)
    image = img_to_array(image) / 127.5 - 1.0  # Normalize to [-1, 1]
    return image

# Loading dataset
def load_dataset(input_path, output_path):
    input_images = sorted(os.listdir(input_path))
    output_images = sorted(os.listdir(output_path))

    inputs, outputs = [], []
    for inp, out in zip(input_images, output_images):
        inp_image = load_and_preprocess_image(os.path.join(input_path, inp))
        out_image = load_and_preprocess_image(os.path.join(output_path, out))
        inputs.append(inp_image)
        outputs.append(out_image)

    return np.array(inputs), np.array(outputs)

# U-Net Generator
def build_generator():
    def down_block(x, filters, apply_batchnorm=True):
        x = Conv2D(filters, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
        if apply_batchnorm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def up_block(x, skip, filters, apply_dropout=False):
        x = Conv2DTranspose(filters, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        if apply_dropout:
            x = Dropout(0.5)(x)
        x = Activation("relu")(x)
        x = Concatenate()([x, skip])
        return x

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    skips = []

    # Encoder (Downsampling)
    x = inputs
    for filters in [64, 128, 256, 512, 512, 512, 512]:
        x = down_block(x, filters, apply_batchnorm=(filters != 64))
        skips.append(x)

    skips = reversed(skips[:-1])

    # Decoder (Upsampling)
    for filters, skip in zip([512, 512, 512, 256, 128, 64], skips):
        x = up_block(x, skip, filters, apply_dropout=(filters == 512))

    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation="tanh")(x)

    return Model(inputs, x)

# PatchGAN Discriminator
def build_discriminator():
    input_img = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    target_img = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = Concatenate()([input_img, target_img])
    for filters in [64, 128, 256, 512]:
        x = Conv2D(filters, kernel_size=4, strides=2, padding="same")(x)
        if filters != 64:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)

    x = Conv2D(1, kernel_size=4, strides=1, padding="same")(x)
    return Model([input_img, target_img], x)

# Loss functions
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (100 * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# Building models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
generator_optimizer = Adam(2e-4, beta_1=0.5)
discriminator_optimizer = Adam(2e-4, beta_1=0.5)

# Checkpoint manager
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

# Training step
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_total_loss, disc_loss

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for input_image, target in dataset:
            gen_loss, disc_loss = train_step(input_image, target)
        print(f"Generator loss: {gen_loss.numpy()}, Discriminator loss: {disc_loss.numpy()}")

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            manager.save()

# Main function
if __name__ == "__main__":
    # Load dataset
    inputs, targets = load_dataset(INPUT_PATH, OUTPUT_PATH)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(BATCH_SIZE)

    # Train the model
    train(dataset, epochs=100)

    # Save the generator model
    generator.save("pix2pix_generator.h5")

    print("Model training complete!")
