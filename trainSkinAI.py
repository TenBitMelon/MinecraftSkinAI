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

# Load a single image
def load_image(filename, input_folder, output_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    input_image = Image.open(input_path)
    output_image = Image.open(output_path)
    return np.array(input_image) / 255.0, np.array(output_image) / 255.0

# Load the dataset using multiprocessing
def load_dataset(input_folder, output_folder):
    filenames = os.listdir(input_folder)
    num_images = len(filenames)
    print(f"Total images: {num_images}")
    
    pool = multiprocessing.Pool()
    results = []
    for filename in filenames:
        if len(results) > 50000:
            break
        result = pool.apply_async(load_image, args=(filename, input_folder, output_folder))
        results.append(result)
    
    input_images = []
    output_images = []
    loaded_count = 0
    for result in results:
        input_image, output_image = result.get()
        input_images.append(input_image)
        output_images.append(output_image)
        loaded_count += 1
        if loaded_count % 1000 == 0:
            print(f"Loaded {loaded_count} images")
    
    pool.close()
    pool.join()
    
    return np.array(input_images), np.array(output_images)

if __name__ == "__main__":
    # Set the GPU memory growth to true
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load the dataset
    start_time_db = time.time()
    print("Loading dataset...")
    input_images, output_images = load_dataset(input_folder, output_folder)
    print(f"Loaded {len(input_images)} images.")

    # Shape the images
    print(f"Shape of array: {input_images.shape}")

    # Create efficient input and output datasets

    # batch_size = 32
    # dataset = tf.data.Dataset.from_tensor_slices((input_images, output_images))
    # dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).repeat().prefetch(1)
    print("Dataset loaded.")
    end_time_db = time.time()
    print(f"Loading dataset took {end_time_db - start_time_db} seconds.")

    # Use the GPU
    with tf.device("/GPU:0"):
        
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
            },
        ]

        for model_config in models:
            start_time_train = time.time()
            print(f"Training {model_config['name']}...")
            model = keras.Sequential(model_config["layers"])
            model.compile(optimizer=model_config["optimizer"], loss=model_config["loss"], metrics=["accuracy"])
            model.fit(input_images, output_images, epochs=100, batch_size=16)

            end_time_train = time.time()
            print(f"Training {model_config['name']} took {end_time_train - start_time_train} seconds.")

            # Save the model
            print(f"Saving {model_config['name']}...")
            model.save(f"{model_config['name']}_image_conversion_model.h5")

            print(f"Evaluating {model_config['name']}...")
            test_loss, test_accuracy = model.evaluate(input_images, output_images)
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
