import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras

# Load the saved model
model = keras.models.load_model('image_conversion_model.h5')

# Create a GUI window
window = tk.Tk()
window.title("Image Conversion")

# Create a canvas to draw on
canvas = tk.Canvas(window, width=64, height=64, bg="white")
canvas.pack(side=tk.LEFT)

# Create a label to display the model output
output_label = tk.Label(window)
output_label.pack(side=tk.RIGHT)

# Store the drawn points
points = []

# Function to handle mouse dragging on the canvas


def on_drag(event):
    x = event.x
    y = event.y
    canvas.create_oval(x, y, x+1, y+1, fill="black", outline="black")
    points.append((x, y))

# Function to handle clearing the canvas


def clear_canvas():
    canvas.delete("all")
    del points[:]

# Function to convert the drawn image and display the model output


def convert_image():
    if len(points) > 0:
        # Create an image from the drawn points
        image = Image.new("RGBA", (64, 64), "white")
        pixels = image.load()
        for point in points:
            x, y = point
            if 0 <= x < 64 and 0 <= y < 64:  # Check if x and y are within valid range
                pixels[x, y] = 0

        # Preprocess the image
        input_image = np.array(image) / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        # Generate the output image using the model
        output_image = model.predict(input_image)
        output_image = np.squeeze(output_image, axis=0)

        # Display the output image
        output_image = Image.fromarray(np.uint8(output_image * 255.0))
        output_image = ImageTk.PhotoImage(output_image)
        output_label.configure(image=output_image)
        output_label.image = output_image


# Bind mouse events to the canvas
canvas.bind("<B1-Motion>", on_drag)

# Create buttons for clearing the canvas and converting the image
clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()
convert_button = tk.Button(window, text="Convert", command=convert_image)
convert_button.pack()

# Start the GUI event loop
window.mainloop()
