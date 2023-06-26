from PIL import Image, ImageChops, ImageFilter
import os
import multiprocessing
import random

dataset_dir = "./downloadedskins"
output_dir = "./preppedskins"
alpha_mask_path = "./skinmask.png"

os.makedirs(output_dir, exist_ok=True)

alpha_mask = Image.open(alpha_mask_path).convert("L")
alpha_mask = alpha_mask.resize((64, 64))

def expand_solid_regions(image):
    expanded_image = Image.new("RGBA", image.size)
    pixels = image.load()
    expanded_pixels = expanded_image.load()
    width, height = image.size

    for x in range(width):
        for y in range(height):
            pixel = pixels[x, y]
            alpha = pixel[3]

            if alpha > 200:
                expanded_pixels[x, y] = pixel
            else:
                neighbors = [
                    pixels[x - 1, y] if x > 0 else (0, 0, 0, 0),
                    pixels[x + 1, y] if x < width - 1 else (0, 0, 0, 0),
                    pixels[x, y - 1] if y > 0 else (0, 0, 0, 0),
                    pixels[x, y + 1] if y < height - 1 else (0, 0, 0, 0),
                ]
                
                # Get the first neighbor with an alpha value greater than 200
                expanded_pixels[x, y] = next((n for n in neighbors if n[3] > 200), (0, 0, 0, 0))

    return expanded_image

def expand_until_solid(image):
    expanded_image = image
    while True:
        expanded_image = expand_solid_regions(expanded_image)
        if expanded_image == image:
            break
        image = expanded_image

    return expanded_image

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

        original_palette = image.getpalette()
        converted_image = expand_until_solid(image)
        converted_image = converted_image.filter(ImageFilter.GaussianBlur(radius=2))
        num_colors = random.randint(2, 5)
        converted_image = converted_image.quantize(palette=original_palette)
        converted_image = converted_image.quantize(colors=num_colors)
        converted_image = converted_image.convert("RGBA")

        converted_image = Image.alpha_composite(image, converted_image)
        converted_image.putalpha(combined_alpha)

        output_path = os.path.join(output_dir, filename)
        converted_image.save(output_path)

        return 1

if __name__ == "__main__":
    image_files = os.listdir(dataset_dir)
    # image_files = image_files[:1000]

    pool = multiprocessing.Pool()

    results = pool.map(process_image, image_files)
    pool.close()
    pool.join()

    processed_count = 0
    for result in results:
        if result is not None:
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"Processed image {processed_count}")
