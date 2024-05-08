import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# Define the generator model
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(latent_dim,)))
    model.add(layers.Dense(7 * 7 * 64, use_bias=False, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    model.add(layers.Reshape((7, 7, 64)))
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh', kernel_regularizer=regularizers.l2(0.02)))
    return model

# Define the discriminator model
def build_discriminator(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    gan_input = tf.keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = tf.keras.Model(gan_input, gan_output)
    return gan

# Load the MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize images to [-1, 1]

# Define the size of the random noise vector
latent_dim = 100

# Build the generator and discriminator models
generator = build_generator(latent_dim)
discriminator = build_discriminator((28, 28, 1))

# Compile the discriminator model
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

# Set discriminator weights to non-trainable
discriminator.trainable = False

# Define the GAN model
gan = build_gan(generator, discriminator)

# Compile the GAN model
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007, beta_1=0.5),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

# Train the GAN model
batch_size = 128
epochs = 50
num_batches = train_images.shape[0] // batch_size

for epoch in range(epochs):
    for batch_idx in range(num_batches):
        # Generate random noise vectors
        noise = tf.random.normal(shape=(batch_size, latent_dim))

        # Generate fake images using the generator
        fake_images = generator.predict(noise)

        # Select a random batch of real images
        real_images = train_images[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # Concatenate real and fake images
        images = tf.concat([real_images, fake_images], axis=0)

        # Create labels for real and fake images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # Add random noise to the labels (important for the stability of training)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        discriminator_loss = discriminator.train_on_batch(images, labels)

        # Generate new random noise vectors
        noise = tf.random.normal(shape=(batch_size, latent_dim))

        # Create labels for fake images (all are "real" to trick the discriminator)
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (via the GAN model)
        gan_loss = gan.train_on_batch(noise, misleading_labels)

    # Print the loss after each epoch
    print(f"Epoch {epoch + 1}, Discriminator Loss: {discriminator_loss}, GAN Loss: {gan_loss}")

# Save the entire model to disk
gan.save('my_model.h5')

# Download the saved model directory to local machine
# from google.colab import files
# files.download('/content/my_model/h5_model')
