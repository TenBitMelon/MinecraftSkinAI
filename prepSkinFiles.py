from PIL import Image, ImageChops, ImageFilter
import os
import multiprocessing
import random

dataset_dir = "./downloadedskins/skins"
output_dir = "./preppedskins"
alpha_mask_path = "./skinmask.png"
alpha_mask_no_layer_path = "./skinmask-nolayer.png"
multiple_images = True # Make only one image per skin (false) or generate multiple images per skin (true)

os.makedirs(output_dir, exist_ok=True)

# Load the alpha mask
alpha_mask = Image.open(multiple_images and alpha_mask_no_layer_path or alpha_mask_path).convert("L")
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

        # Get the original alpha channel and convert it to a mask
        original_alpha = image.getchannel("A")
        original_alpha = original_alpha.point(lambda p: 0 if p < 200 else 255, mode="L")

        # Combine the original alpha channel with the blank skin mask
        combined_alpha = ImageChops.darker(original_alpha, alpha_mask)

        if not multiple_images:
            converted_image = expand_until_solid(image)
            converted_image = converted_image.filter(ImageFilter.GaussianBlur(radius=2))
            num_colors = random.randint(2, 5)
            converted_image = converted_image.quantize(colors=num_colors)
            converted_image = converted_image.convert("RGBA")

            converted_image = Image.alpha_composite(image, converted_image)
            converted_image.putalpha(combined_alpha)

            output_path = os.path.join(output_dir, filename)
            converted_image.save(output_path)
        else:
            converted_image = image
            converted_image.putalpha(combined_alpha)
            converted_image = expand_until_solid(converted_image)

            blur_values = [0, 2, 4]
            color_values = [2, 3, 4, 5]
            for blur in blur_values:
                # Blur the image
                blur_image = converted_image.filter(ImageFilter.GaussianBlur(radius=blur))
                for color in color_values:
                    # Quantize the image
                    colored_image = blur_image.quantize(colors=color).convert("RGBA")
                    # Blur the image again
                    blur_color_image = colored_image.filter(ImageFilter.GaussianBlur(radius=2))

                    colored_image.putalpha(combined_alpha)
                    blur_color_image.putalpha(combined_alpha)

                    # Save the image
                    snipped_filename = filename[:-4]

                    new_filename = snipped_filename + f"_b{blur}_c{color}.png"
                    output_path = os.path.join(output_dir, new_filename)
                    colored_image.save(output_path)

                    new_filename = snipped_filename + f"_b{blur}_c{color}_b.png"
                    output_path = os.path.join(output_dir, new_filename)
                    blur_color_image.save(output_path)

        return 1

if __name__ == "__main__":
    image_files = os.listdir(dataset_dir)
    # image_files = image_files[:1000]

    pool = multiprocessing.Pool()

    processed_count = 0
    for result in pool.map(process_image, image_files):
        if result is not None:
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count} images")

    pool.close()
    pool.join()
