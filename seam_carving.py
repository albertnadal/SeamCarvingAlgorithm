import numpy as np
from PIL import Image
from typing import List
from typing import Optional
from typing import Tuple

import math
import sys

DESIRED_IMAGE_WIDTH = 1348

image = Image.open("original.jpg")
image = image.convert("RGB")

assert (
    image.width > DESIRED_IMAGE_WIDTH
), f"The width of the original image ({image.width}) must be higher than the DESIRED_IMAGE_WIDTH ({DESIRED_IMAGE_WIDTH})."

sys.setrecursionlimit(image.height + 10)

# Convert the image to grayscale
grayscale_image = Image.new("RGB", image.size)
grayscale_pixels = []
for pixel in image.getdata():
    r, g, b = pixel
    avg_color = int(0.299 * r + 0.587 * g + 0.114 * b)
    grayscale_pixels.append((avg_color, avg_color, avg_color))
grayscale_image.putdata(grayscale_pixels)
grayscale_image.save("grayscale.jpg", quality=100)

# Generate and initialize the energy map by using the Sobel filter on the grayscale image
energy_map = np.zeros((grayscale_image.width, grayscale_image.height), dtype=object)
for x in range(grayscale_image.width):
    for y in range(grayscale_image.height):
        energy_map[x, y] = {"energy": 0, "sum": None, "directions": []}


def get_pixel(image: Image, x: int, y: int) -> Tuple[int, int, int]:
    x = max(0, min(x, image.width - 1))
    y = max(0, min(y, image.height - 1))
    return image.getpixel((x, y))


max_magnitude = 0
for x in range(grayscale_image.width):
    for y in range(grayscale_image.height):
        top_left_pixel = get_pixel(grayscale_image, x - 1, y - 1)[0]
        top_pixel = get_pixel(grayscale_image, x, y - 1)[0]
        top_right_pixel = get_pixel(grayscale_image, x + 1, y - 1)[0]
        left_pixel = get_pixel(grayscale_image, x - 1, y)[0]
        right_pixel = get_pixel(grayscale_image, x + 1, y)[0]
        bottom_left_pixel = get_pixel(grayscale_image, x - 1, y + 1)[0]
        bottom_pixel = get_pixel(grayscale_image, x, y + 1)[0]
        bottom_right_pixel = get_pixel(grayscale_image, x + 1, y + 1)[0]

        hor_kernel_result = (
            top_left_pixel * -1
            + top_right_pixel
            + left_pixel * -2
            + right_pixel * 2
            + bottom_left_pixel * -1
            + bottom_right_pixel
        )

        ver_kernel_result = (
            top_left_pixel
            + top_pixel * 2
            + top_right_pixel
            + bottom_left_pixel * -1
            + bottom_pixel * -2
            + bottom_right_pixel * -1
        )

        magnitude = math.sqrt(hor_kernel_result**2 + ver_kernel_result**2)
        energy_map[x, y]["energy"] = magnitude
        if y == 0:
            energy_map[x, y]["sum"] = magnitude
        max_magnitude = max(max_magnitude, magnitude)

# Save the initial energy map in a image file
energy_image = Image.new("RGB", (image.width, image.height))
for x in range(image.width):
    for y in range(image.height):
        color_component = (
            0
            if max_magnitude == 0
            else int((energy_map[x][y]["energy"] * 255) // max_magnitude)
        )
        energy_image.putpixel(
            (x, y), (color_component, color_component, color_component)
        )
energy_image.save("energy.jpg", quality=100)

# Start searching for seams and crop the image one pixel width on each iteration
width = image.width
num_pixels_to_crop = width - DESIRED_IMAGE_WIDTH

for iteration in range(num_pixels_to_crop):
    seams_found: List[Tuple[int, List[Tuple[int, int]]]] = []

    # Calculate seam paths in the energy map
    for y in range(image.height - 1):
        for x in range(width):
            if x > 0:
                current_bottom_left_sum = energy_map[x - 1, y + 1]["sum"]
                bottom_left_energy_sum = (
                    energy_map[x - 1, y + 1]["energy"] + energy_map[x, y]["sum"]
                )
                if (
                    current_bottom_left_sum is None
                    or current_bottom_left_sum > bottom_left_energy_sum
                ):
                    energy_map[x - 1, y + 1]["sum"] = bottom_left_energy_sum
                    energy_map[x, y]["directions"].append(-1)
                    if 0 in energy_map[x - 1, y]["directions"]:
                        energy_map[x - 1, y]["directions"].remove(0)

                    if x > 1 and 1 in energy_map[x - 2, y]["directions"]:
                        energy_map[x - 2, y]["directions"].remove(1)

            current_bottom_sum = energy_map[x, y + 1]["sum"]
            bottom_energy_sum = energy_map[x, y + 1]["energy"] + energy_map[x, y]["sum"]
            if current_bottom_sum is None or current_bottom_sum >= bottom_energy_sum:
                energy_map[x, y + 1]["sum"] = bottom_energy_sum
                energy_map[x, y]["directions"].append(0)
                if x > 0 and 1 in energy_map[x - 1, y]["directions"]:
                    energy_map[x - 1, y]["directions"].remove(1)

            if x <= width - 2:
                energy_map[x + 1, y + 1]["sum"] = (
                    energy_map[x + 1, y + 1]["energy"] + energy_map[x, y]["sum"]
                )
                energy_map[x, y]["directions"].append(1)

    # Get the seams with the lowest energy
    def get_seam_at_position(x: int, y: int = 0) -> Optional[Tuple[int, List[int]]]:
        if y == image.height - 1:
            return energy_map[x, y]["sum"], [x]

        if not energy_map[x, y]["directions"]:
            return None

        best_sub_seam: List[int] = None
        lowest_energy: int = None

        for direction_delta in energy_map[x, y]["directions"]:
            sub_seam = get_seam_at_position(x=x + direction_delta, y=y + 1)
            if sub_seam and (lowest_energy is None or sub_seam[0] < lowest_energy):
                lowest_energy = sub_seam[0]
                best_sub_seam = sub_seam[1]

        return (lowest_energy, [x] + best_sub_seam) if best_sub_seam else None

    # Get the seam for each x position of the image
    for x in range(width):
        seam: Optional[Tuple[int, List[int]]] = get_seam_at_position(x)
        if seam is not None:
            seams_found.append(seam)

    # Sort seams according to the energy
    sorted_seams = sorted(seams_found, key=lambda x: x[0])

    # Get the seam with the lowest energy
    best_seam = sorted_seams[0]

    # Crop the image and the energy map by removing the pixels of the selected seam
    width -= 1
    cropped_image = Image.new("RGB", (width, image.height))
    new_energy_map = np.zeros((width, grayscale_image.height), dtype=object)

    for x in range(width):
        for y in range(grayscale_image.height):
            new_energy_map[x, y] = {"energy": 0, "sum": None, "directions": []}

    for y in range(image.height):
        end_x = min(width, best_seam[1][y])
        for x in range(0, end_x):
            cropped_image.putpixel((x, y), image.getpixel((x, y)))
            new_energy_map[x, y]["energy"] = energy_map[x, y]["energy"]
            if y == 0:
                new_energy_map[x, y]["sum"] = energy_map[x, y]["energy"]

        for x in range(end_x, width):
            cropped_image.putpixel((x, y), image.getpixel((x + 1, y)))
            new_energy_map[x, y]["energy"] = energy_map[x + 1, y]["energy"]
            if y == 0:
                new_energy_map[x, y]["sum"] = energy_map[x + 1, y]["energy"]

    image = cropped_image
    energy_map = new_energy_map
    image.save(f"cropped_{iteration}.jpg", quality=100)
