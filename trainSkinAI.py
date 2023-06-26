import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import multiprocessing
import time

# Set the paths to your input and output image folders
input_folder = "./preppedskins"
output_folder = "./downloadedskins"

# Set the image size
image_size = (64, 64)

def load_image(filename, input_folder, output_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    input_image = Image.open(input_path)
    output_image = Image.open(output_path)
    return np.array(input_image) / 255.0, np.array(output_image) / 255.0

# Generator function to load images in batches
def image_generator(filenames, input_folder, output_folder, batch_size):
    for filename in filenames:
        input_image, output_image = load_image(filename, input_folder, output_folder)
        yield input_image, output_image

# Load the dataset using multiprocessing
def load_dataset(input_folder, output_folder, batch_size):
    filenames = os.listdir(input_folder)
    num_images = len(filenames)
    print(f"Total images: {num_images}")

    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(filenames, input_folder, output_folder, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(image_size[0], image_size[1], 4), dtype=tf.float32),
            tf.TensorSpec(shape=(image_size[0], image_size[1], 4), dtype=tf.float32),
        )
    )
    
    return num_images, dataset

if __name__ == "__main__":
    # Set the GPU memory growth to true
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Load the dataset
    start_time_db = time.time()
    print("Loading dataset...")
    batch_size = 32
    num_loaded_images, dataset = load_dataset(input_folder, output_folder, batch_size)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).repeat().prefetch(1)
    
    input_images, output_images = next(iter(dataset))
    print(f"Loaded {len(input_images)} images.")
    print(f"Shape of array: {input_images.shape}")
    print("Dataset loaded.")
    
    end_time_db = time.time()
    print(f"Loading dataset took {end_time_db - start_time_db} seconds.")

    # Use the GPU
    with tf.device("/GPU:0"), tf.distribute.OneDeviceStrategy("/GPU:0").scope():
        
        
        # Create the model
        models = [
            # {
            #     "name": "The_Van_Gogh_Ultima",
            #     "layers": [
            #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(image_size[0], image_size[1], 4)),
            #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            #         keras.layers.BatchNormalization(),
            #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            #         keras.layers.BatchNormalization(),
            #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
            #     ],
            #     "optimizer": "adam",
            #     "loss": "mean_squared_error",
            # },
            {
                "name": "The_Van_Gogh_Enigma",
                "layers": [
                    keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(image_size[0], image_size[1], 4)),
                    keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                    keras.layers.BatchNormalization(),
                    keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                    keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                    keras.layers.BatchNormalization(),
                    keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                    keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                    keras.layers.BatchNormalization(),
                    keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                    keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                    keras.layers.BatchNormalization(),
                    # keras.layers.Dropout(0.5),
                    # keras.layers.Dense(512, activation="relu"),
                    # keras.layers.Dropout(0.5),
                    keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
                ],
                "optimizer": "adam",
                "loss": "mean_squared_error",
                # "loss": "categorical_crossentropy",
            },
            {
                "name": "The_Van_Gogh_Optimistic", # Diffusion based model done by using a more complex model
                "layers": [
                    keras.layers.Input(shape=(image_size[0], image_size[1], 4)),
                ],
            }
        ]

        for model_config in models:
            start_time_train = time.time()
            print(f"Training {model_config['name']}...")

            model = tf.keras.models.load_model(f"{model_config['name']}_image_conversion_model.h5")

            if model is None:
                model = keras.Sequential(model_config["layers"])
                model.compile(optimizer=model_config["optimizer"], loss=model_config["loss"], metrics=["accuracy", "mean_squared_error"])

            callbacks = [
                keras.callbacks.ModelCheckpoint(f"{model_config['name']}_checkpoint.h5", save_best_only=True, monitor="loss"),
                keras.callbacks.EarlyStopping(monitor="loss", patience=10),
            ]

            history = model.fit(dataset, epochs=110, steps_per_epoch=num_loaded_images // batch_size, callbacks=callbacks)

            # history = model.fit(dataset, epochs=100, steps_per_epoch=num_loaded_images // batch_size, callbacks=callbacks, validation_split=validation_split, validation_steps=validation_steps)

            end_time_train = time.time()
            print(f"Training {model_config['name']} took {end_time_train - start_time_train} seconds.")

            # Save the model
            print(f"Saving {model_config['name']}...")
            model.save(f"{model_config['name']}_image_conversion_model.h5")

            print(f"Evaluating {model_config['name']}...")
            test_loss, test_accuracy = model.evaluate(dataset, steps=1)
            print(f"{model_config['name']} - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# # Build the model
# # model = keras.Sequential([
# #     keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
# #                         input_shape=(image_size[0], image_size[1], 4)),
# #     keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
# #     keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
# #     keras.layers.Conv2D(4, (3, 3), activation='sigmoid', padding='same')
# # ])

# # Compile the model
# # model.compile(optimizer='adam', loss='binary_crossentropy',
# #               metrics=['accuracy'])

# # # Train the model
# # model.fit(input_images, output_images, epochs=100, batch_size=32)

# # # Save the model
# # model.save('image_conversion_model.h5')

# # List of model configurations with witty names
# model_configs = [
# # {
# #     "name": "The Picasso",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             32,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "binary_crossentropy",
# # },
# # {
# #     "name": "The Monet",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             64,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "binary_crossentropy",
# # },
# # {
# #     "name": "The Dali",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             32,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "rmsprop",
# #     "loss": "mean_squared_error",
# # },
# # {
# #     "name": "The Warhol",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             64,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "categorical_crossentropy",
# # },
# # {
# #     "name": "The Van Gogh",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             32,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "mean_squared_error",
# # },
# # {
# #     "name": "The Rembrandt",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             32,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "binary_crossentropy",
# # },
# # {
# #     "name": "The Banksy",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             64,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "mean_squared_error",
# # },
# # {
# #     "name": "The Warhol",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             32,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "categorical_crossentropy",
# # },
# # {
# #     "name": "The Van Gogh Enhanced",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             64,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "mean_squared_error",
# # },
# # {
# #     "name": "The Van Gogh Advanced",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             32,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.BatchNormalization(),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.BatchNormalization(),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.BatchNormalization(),
# #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# #         keras.layers.BatchNormalization(),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "mean_squared_error",
# # },
# # {
# #     "name": "The Van Gogh Expressive",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             64,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "mean_squared_error",
# # },
# # {
# #     "name": "The Van Gogh Specialized",
# #     "layers": [
# #         keras.layers.Conv2D(
# #             32,
# #             (3, 3),
# #             activation="relu",
# #             padding="same",
# #             input_shape=(image_size[0], image_size[1], 4),
# #         ),
# #         keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# #         keras.layers.Conv2D(4, (1, 1), activation="softmax", padding="valid"),
# #     ],
# #     "optimizer": "adam",
# #     "loss": "mean_squared_error",
# # },
# {
# "name": "The Van Gogh Ultima",
# "layers": [
# keras.layers.Conv2D(
# 64,
# (3, 3),
# activation="relu",
# padding="same",
# input_shape=(image_size[0], image_size[1], 4),
# ),
# keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
# keras.layers.BatchNormalization(),
# keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
# keras.layers.BatchNormalization(),
# keras.layers.Conv2D(4, (3, 3), activation="softmax", padding="same"),
# ],
# "optimizer": "adam",
# "loss": "mean_squared_error",
# }
# ]
