from PIL import Image, ImageChops
import os

dataset_dir = "./downloadedskins/skins"
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

        # Everywhere the alpha mask is black, make the image's pixel black too
        for x in range(64):
            for y in range(64):
                if combined_alpha.getpixel((x, y)) == 0:
                    image.putpixel((x, y), (0, 0, 0, 0))

        output_path = os.path.join(dataset_dir, filename)
        image.save(output_path)

        return 1

image_files = os.listdir(dataset_dir)

if __name__ == "__main__":
    # Multiprocessing
    from multiprocessing import Pool
    with Pool(8) as p:
        print(p.map(process_image, image_files))