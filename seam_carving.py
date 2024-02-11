from PIL import Image
from typing import List
from typing import Optional
from typing import Tuple

import math
import sys

image = Image.open("original.jpg")
image = image.convert('RGB')

sys.setrecursionlimit(image.height + 10)

def getpixel(image: Image, x: int, y: int) -> Tuple[int]:
    x = 0 if x < 0 else min(x, image.width)
    x = image.width - 1 if x >= image.width else x
    y = 0 if y < 0 else min(y, image.height)
    y = image.height -1 if y >= image.height else y
    return image.getpixel((x, y))

# Convert image to gray scale
grayscale_image = Image.new("RGB", (image.width, image.height))
for x in range(image.width):
    for y in range(image.height):
        pixel = image.getpixel((x, y))
        r, g, b = pixel
        avg_color = int(0.299*r + 0.587*g + 0.114*b)
        grayscale_image.putpixel((x, y), (avg_color, avg_color, avg_color))

grayscale_image.save("grayscale.jpg", quality=100)

# Calculate the energy map by using the Sobel filter on the grayscale image
energy_map = [[{"energy": 0, "sum": None, "directions": []} for _ in range(grayscale_image.height)] for _ in range(grayscale_image.width)]
max_magnitude = 0

for x in range(grayscale_image.width):
    for y in range(grayscale_image.height):
        hor_kernel_result = getpixel(grayscale_image, x - 1, y - 1)[0] * -1  # top-left pixel
        hor_kernel_result += getpixel(grayscale_image, x + 1, y - 1)[0]  # top-right pixel
        hor_kernel_result += getpixel(grayscale_image, x - 1, y)[0] * -2  # left pixel
        hor_kernel_result += getpixel(grayscale_image, x + 1, y)[0] * 2  # right pixel
        hor_kernel_result += getpixel(grayscale_image, x - 1, y + 1)[0] * -1  # bottom-left pixel
        hor_kernel_result += getpixel(grayscale_image, x + 1, y + 1)[0]  # bottom-right pixel

        ver_kernel_result = getpixel(grayscale_image, x - 1, y - 1)[0]  # top-left pixel
        ver_kernel_result += getpixel(grayscale_image, x, y - 1)[0] * 2  # top pixel
        ver_kernel_result += getpixel(grayscale_image, x + 1, y - 1)[0]  # top-right pixel
        ver_kernel_result += getpixel(grayscale_image, x - 1, y + 1)[0] * -1 # bottom-left pixel
        ver_kernel_result += getpixel(grayscale_image, x, y + 1)[0] * -2 # bottom pixel
        ver_kernel_result += getpixel(grayscale_image, x + 1, y + 1)[0] * -1 # bottom-right pixel

        magnitude = math.sqrt(hor_kernel_result**2 + ver_kernel_result**2)
        energy_map[x][y]["energy"] = magnitude
        if y == 0:
            energy_map[x][y]["sum"] = magnitude
        max_magnitude = max(max_magnitude, magnitude)

energy_image = Image.new("RGB", (image.width, image.height))

# Save the initial energy map in a image file
for x in range(image.width):
    for y in range(image.height):
        color_component = 0 if max_magnitude == 0 else int((energy_map[x][y]["energy"] * 255) // max_magnitude)
        energy_image.putpixel((x, y), (color_component, color_component, color_component))

energy_image.save("energy.jpg", quality=100)

width = image.width
total_hor_pixels_to_crop = 400

for iteration in range(total_hor_pixels_to_crop):
    seams_found: List[Tuple[int, List[Tuple[int, int]]]] = []

    # Calculate seam paths in the energy map
    for y in range(image.height - 1):
        for x in range(width):
            if x > 0:
                current_bottom_left_sum = energy_map[x-1][y+1]["sum"]
                bottom_left_energy_sum = energy_map[x-1][y+1]["energy"] + energy_map[x][y]["sum"]
                if (current_bottom_left_sum is None or current_bottom_left_sum > bottom_left_energy_sum):
                    energy_map[x-1][y+1]["sum"] = bottom_left_energy_sum
                    energy_map[x][y]["directions"].append(-1)
                    if 0 in energy_map[x-1][y]["directions"]:
                        energy_map[x-1][y]["directions"].remove(0)

                    if x > 1 and 1 in energy_map[x-2][y]["directions"]:
                        energy_map[x-2][y]["directions"].remove(1)

            current_bottom_sum = energy_map[x][y+1]["sum"]
            bottom_energy_sum = energy_map[x][y+1]["energy"] + energy_map[x][y]["sum"]
            if current_bottom_sum is None or current_bottom_sum >= bottom_energy_sum:
                energy_map[x][y+1]["sum"] = bottom_energy_sum
                energy_map[x][y]["directions"].append(0)
                if x > 0 and 1 in energy_map[x-1][y]["directions"]:
                    energy_map[x-1][y]["directions"].remove(1)

            if x <= width - 2:
                energy_map[x+1][y+1]["sum"] = energy_map[x+1][y+1]["energy"] + energy_map[x][y]["sum"]
                energy_map[x][y]["directions"].append(1)

    # Get seams with the lowest energy
    def get_seam_at_position(x: int, y: int = 0) -> Optional[Tuple[int, List[int]]]:
        if y == image.height - 1:
            return (energy_map[x][y]["sum"], [x])

        if not len(energy_map[x][y]["directions"]):
            return None

        best_sub_seam: List[int] = None
        lowest_energy: int = None

        for direction_delta in energy_map[x][y]["directions"]:
            sub_seam: Optional[Tuple[int, List[int]]] = get_seam_at_position(x=x+direction_delta, y=y+1)
            if sub_seam is not None:
                if lowest_energy is None or sub_seam[0] < lowest_energy:
                    lowest_energy = sub_seam[0]
                    best_sub_seam = sub_seam[1]

        if best_sub_seam is None:
            return None

        return (lowest_energy, [x] + best_sub_seam)

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
    width-=1
    new_energy_map = [[{"energy": 0, "sum": None, "directions": []} for _ in range(image.height)] for _ in range(width)]
    cropped_image = Image.new("RGB", (width, image.height))

    for y in range(image.height):
        end_x = min(width, best_seam[1][y])
        for x in range(0, end_x):
            cropped_image.putpixel((x, y), image.getpixel((x, y)))
            new_energy_map[x][y]["energy"] = energy_map[x][y]["energy"]
            if y == 0:
                new_energy_map[x][y]["sum"] = energy_map[x][y]["energy"]

        for x in range(end_x, width):
            cropped_image.putpixel((x, y), image.getpixel((x+1, y)))
            new_energy_map[x][y]["energy"] = energy_map[x+1][y]["energy"]
            if y == 0:
                new_energy_map[x][y]["sum"] = energy_map[x+1][y]["energy"]

    image = cropped_image
    energy_map = new_energy_map

    image.save(f"cropped_{iteration}.jpg", quality=100)
