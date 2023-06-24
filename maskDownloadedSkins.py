from PIL import Image, ImageChops
import os

dataset_dir = "./downloadedskins"
alpha_mask_path = "./skinmask.png"

alpha_mask = Image.open(alpha_mask_path).convert("L")
alpha_mask = alpha_mask.resize((64, 64))

def process_image(filename):
    if filename.endswith(".png"):
        image_path = os.path.join(dataset_dir, filename)
        image = Image.open(image_path).convert("RGBA")
        if image.size != (64, 64):
            os.remove(image_path)
            return

        original_alpha = image.getchannel("A")
        original_alpha = original_alpha.point(lambda p: 0 if p < 200 else 255, mode="L")

        combined_alpha = ImageChops.darker(original_alpha, alpha_mask)
        image.putalpha(combined_alpha)

        output_path = os.path.join(dataset_dir, filename)
        image.save(output_path)

        return 1

image_files = os.listdir(dataset_dir)
proccessed_images = 0
for image_file in image_files:
    process_image(image_file)
    proccessed_images += 1
    if proccessed_images % 1000 == 0:
        print(f"Processed {proccessed_images} images")