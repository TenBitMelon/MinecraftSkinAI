import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import tkinter as tk
from tkinter import PhotoImage, Label, Scrollbar, Canvas, Frame

testing_folder = "./preppedskins"
# amounttesting = len(os.listdir(testing_folder))
amounttesting = 50
output_folder = "./skins"

# Load the saved model
# model = keras.models.load_model("The Banksy_image_conversion_model.h5") +
# model = keras.models.load_model("The Dali_image_conversion_model.h5")
# model = keras.models.load_model("The Monet_image_conversion_model.h5") trash
# model = keras.models.load_model("The Picasso_image_conversion_model.h5") trash
# model = keras.models.load_model("The Rembrandt_image_conversion_model.h5") trash
# model = keras.models.load_model("The Van Gogh_image_conversion_model.h5") + best
# model = keras.models.load_model("The Warhol_image_conversion_model.h5")
# model = keras.models.load_model("The Van Gogh Ultima_image_conversion_model.h5")
model = keras.models.load_model("The_Van_Gogh_Enigma_image_conversion_model.h5")

# Get the list of files in the testing folder
filenames = os.listdir(testing_folder)

# Create a GUI window
window = tk.Tk()
window.title("Image Display")
window.geometry("1200x600")

# Create a frame for the canvas and scrollbar
frame = Frame(window)
frame.pack(side="left", fill="both", expand=True)

# Create a canvas with a scrollbar
canvas = Canvas(frame)
scrollbar = Scrollbar(frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create a frame to hold the images
image_frame = Frame(canvas)
canvas.create_window((0, 0), window=image_frame, anchor="nw")

# Process and display each image
for i, filename in enumerate(filenames):
    # Stop after the specified amount of images
    if i >= amounttesting:
        break

    # Get the image from the file
    image = (
        Image.open(os.path.join(testing_folder, filename))
        .convert("RGBA")
        .resize((64, 64))
    )

    # Preprocess the image
    input_image = np.array(image) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    output_image = input_image

    # Loop the image through the model
    for _ in range(1):
        # Generate the output image using the model
        output_image = model.predict(output_image)

    output_image = np.squeeze(output_image, axis=0)

    # Clamp the alpha channel to be either 0 or 1
    # output_image[..., 3] = np.where(output_image[..., 3] > 0.3, 1.0, 0.0)

    # Save the output image
    output_image = Image.fromarray(np.uint8(output_image * 255.0))
    output_image_path = os.path.join(output_folder, f"output_{i}.png")
    input_image_path = os.path.join(testing_folder, filename)
    output_image.save(output_image_path)

    # Load the images into Tkinter PhotoImage
    input_photo = PhotoImage(file=input_image_path)
    output_photo = PhotoImage(file=output_image_path)

    # Resize the images to be larger
    input_photo = input_photo.zoom(5, 5)
    output_photo = output_photo.zoom(5, 5)

    # Create labels to display the images side by side then the next below them
    input_label = Label(image_frame, image=input_photo)
    input_label.image = input_photo
    input_label.grid(row=i // 5, column=i % 5 * 2)

    output_label = Label(image_frame, image=output_photo)
    output_label.image = output_photo
    output_label.grid(row=i // 5, column=i % 5 * 2 + 1)


# Configure the canvas to fit the frame size
image_frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

# Run the GUI event loop
window.mainloop()
