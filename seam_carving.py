from PIL import Image
from typing import Tuple

import math

image = Image.open("original.jpg")
pixels = image.load()

def getpixel(image: Image, x: int, y: int) -> Tuple[int]:
    x = 0 if x < 0 else min(x, image.width)
    x = image.width - 1 if x >= image.width else x
    y = 0 if y < 0 else min(y, image.height)
    y = image.height -1 if y >= image.height else y
    return image.getpixel((x, y))

# Convert image to gray scale
for x in range(image.width):
    for y in range(image.height):
        pixel = image.getpixel((x, y))
        r, g, b = pixel
        avg_color = int(0.299*r + 0.587*g + 0.114*b)
        image.putpixel((x, y), (avg_color, avg_color, avg_color))

image.save("grayscale.jpg")

# Calculate the energy map by using the Sobel filter
energy_map = [[{"energy": 0, "sum": None, "directions": []} for _ in range(image.height)] for _ in range(image.width)]
max_magnitude = 0

for x in range(image.width):
    for y in range(image.height):
        hor_kernel_result = getpixel(image, x - 1, y - 1)[0] * -1  # top-left pixel
        hor_kernel_result += getpixel(image, x + 1, y - 1)[0]  # top-right pixel
        hor_kernel_result += getpixel(image, x - 1, y)[0] * -2  # left pixel
        hor_kernel_result += getpixel(image, x + 1, y)[0] * 2  # right pixel
        hor_kernel_result += getpixel(image, x - 1, y + 1)[0] * -1  # bottom-left pixel
        hor_kernel_result += getpixel(image, x + 1, y + 1)[0]  # bottom-right pixel

        ver_kernel_result = getpixel(image, x - 1, y - 1)[0]  # top-left pixel
        ver_kernel_result += getpixel(image, x, y - 1)[0] * 2  # top pixel
        ver_kernel_result += getpixel(image, x + 1, y - 1)[0]  # top-right pixel
        ver_kernel_result += getpixel(image, x - 1, y + 1)[0] * -1 # bottom-left pixel
        ver_kernel_result += getpixel(image, x, y + 1)[0] * -2 # bottom pixel
        ver_kernel_result += getpixel(image, x + 1, y + 1)[0] * -1 # bottom-right pixel

        magnitude = math.sqrt(hor_kernel_result**2 + ver_kernel_result**2)
        energy_map[x][y]["energy"] = magnitude
        if y == 0:
            energy_map[x][y]["sum"] = magnitude
        max_magnitude = max(max_magnitude, magnitude)

energy_image = Image.new("RGB", (image.width, image.height))

# Save the energy map in a image file
for x in range(image.width):
    for y in range(image.height):
        color_component = int((energy_map[x][y]["energy"] * 255) // max_magnitude)
        energy_image.putpixel((x, y), (color_component, color_component, color_component))

energy_image.save("energy.jpg")

# Find seams in the energy map
for y in range(image.height - 1):
    for x in range(image.width):
        if x > 0:
            current_bottom_left_sum = energy_map[x-1][y+1]["sum"]
            bottom_left_energy_sum = energy_map[x-1][y+1]["energy"] + energy_map[x][y]["energy"]
            if current_bottom_left_sum > bottom_left_energy_sum:
                energy_map[x-1][y+1]["sum"] = bottom_left_energy_sum
                energy_map[x][y]["directions"].append(-1)
                if 0 in energy_map[x-1][y]["directions"]:
                    energy_map[x-1][y]["directions"].remove(0)

                if x > 1 and 1 in energy_map[x-2][y]["directions"]:
                    energy_map[x-2][y]["directions"].remove(1)

        current_bottom_sum = energy_map[x][y+1]["sum"]
        bottom_energy_sum = energy_map[x][y+1]["energy"] + energy_map[x][y]["energy"]
        if current_bottom_sum is None or current_bottom_sum > bottom_energy_sum:
            energy_map[x][y+1]["sum"] = bottom_energy_sum
            energy_map[x][y]["directions"].append(0)
            if x > 0 and 1 in energy_map[x-1][y]["directions"]:
                energy_map[x-1][y]["directions"].remove(1)

        if x < image.width - 2:
            energy_map[x+1][y+1]["sum"] = energy_map[x+1][y+1]["energy"] + energy_map[x][y]["energy"]
            energy_map[x][y]["directions"].append(1)
