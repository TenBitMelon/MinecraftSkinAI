import os
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

class ImageViewer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = self.get_image_files()
        self.current_image_index = 0

        self.root = Tk()
        self.root.title("Image Viewer")
        self.root.geometry("500x500")
        self.root.bind("<Left>", lambda event: self.next_image())
        self.root.bind("<Right>", lambda event: self.delete_image())

        self.canvas = Canvas(self.root, width=400, height=400)
        self.canvas.pack()

        self.label = Label(self.root, text="")
        self.label.pack()

        self.yes_button = Button(self.root, text="Yes", command=self.next_image)
        self.yes_button.pack(side=LEFT, padx=10, pady=10)

        self.no_button = Button(self.root, text="No", command=self.delete_image)
        self.no_button.pack(side=LEFT, padx=10, pady=10)

        self.load_image()

    def get_image_files(self):
        image_files = []
        for file_name in os.listdir(self.folder_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                image_files.append(os.path.join(self.folder_path, file_name))
        return image_files

    def load_image(self):
        if self.current_image_index < len(self.image_files):
            image_file = self.image_files[self.current_image_index]
            image = Image.open(image_file)
            # Resize the image to fit the canvas without antialiasing
            image = image.resize((400, 400), Image.NEAREST)
            photo = ImageTk.PhotoImage(image)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=photo)
            self.canvas.image = photo

            self.label.config(text=f"Image {self.current_image_index + 1}/{len(self.image_files)}")
        else:
            messagebox.showinfo("No more images", "You have viewed all the images.")

    def next_image(self):
        self.current_image_index += 1
        self.load_image()

    def delete_image(self):
        if self.current_image_index < len(self.image_files):
            image_file = self.image_files[self.current_image_index]
            os.remove(image_file)
            self.image_files.pop(self.current_image_index)
            self.load_image()
        else:
            messagebox.showinfo("No more images", "You have viewed all the images.")

if __name__ == '__main__':
    folder_path = input("Enter the folder path: ")
    image_viewer = ImageViewer(folder_path)
    image_viewer.root.mainloop()
