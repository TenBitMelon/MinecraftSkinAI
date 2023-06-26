import tkinter as tk
from tkinter import filedialog, colorchooser
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras

# Load the saved model
model = keras.models.load_model('The_Van_Gogh_Enigma_image_conversion_model.h5')
# model = keras.models.load_model('image_conversion_model.h5')

# Create a GUI window
window = tk.Tk()
window.title("Image Conversion")

# Create a canvas to draw on with a off-white background
canvas = tk.Canvas(window, width=64, height=64, bg="#f0f0f0")
canvas.pack(side=tk.LEFT)

# Create a label to display the model output
output_label = tk.Label(window)
output_label.pack(side=tk.RIGHT)

# Store the drawn points
points = []
selected_color = ((0, 0, 0, 255), "#000000")

# Function to handle mouse dragging on the canvas
def on_drag(event):
    x = event.x
    y = event.y
    canvas.create_oval(x, y, x+1, y+1, fill=selected_color[1], outline=selected_color[1])
    points.append((x, y, selected_color[0]))

# Function to handle clearing the canvas

def clear_canvas():
    canvas.delete("all")
    del points[:]

# Function to convert the drawn image and display the model output
def convert_image():
    if len(points) > 0:
        # Create an image from the drawn points
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        pixels = image.load()
        for point in points:
            x, y, color = point
            if 0 <= x < 64 and 0 <= y < 64:  # Check if x and y are within valid range
                pixels[x, y] = color # Set the pixel color

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

# Function to handle color selection
def select_color():
    global selected_color
    selected_color = colorchooser.askcolor()

# Function to handle file selection
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if file_path:
        clear_canvas()
        image = Image.open(file_path).resize((64, 64), Image.ANTIALIAS)
        pixels = image.load()
        for i in range(image.width):
            for j in range(image.height):
                if pixels[i, j][3] > 0:  # Check if the pixel is not fully transparent
                    canvas.create_oval(i, j, i+1, j+1, fill="#%02x%02x%02x" % pixels[i, j][:3], outline="#%02x%02x%02x" % pixels[i, j][:3])
                    points.append((i, j, pixels[i, j]))

# Bind mouse events to the canvas
canvas.bind("<B1-Motion>", on_drag)

# Create buttons for clearing the canvas and converting the image
clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()
convert_button = tk.Button(window, text="Convert", command=convert_image)
convert_button.pack()

# Create buttons for color selection and file selection
color_button = tk.Button(window, text="Select Color", command=select_color)
color_button.pack()
file_button = tk.Button(window, text="Select File", command=select_file)
file_button.pack()

# Start the GUI event loop
window.mainloop()
