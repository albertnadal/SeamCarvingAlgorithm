from PIL import Image
import math

THRESHOLD = 720

image = Image.open("original.jpg")
pixels = image.load()

# Convert image to gray scale
for x in range(image.width):
    for y in range(image.height):
        pixel = image.getpixel((x, y))
        r, g, b = pixel
        avg_color = int(0.299*r + 0.587*g + 0.114*b)
        image.putpixel((x, y), (avg_color, avg_color, avg_color))

image.save("grayscale.jpg")

# Calculate the energy map by using the Sobel filter
energy_map = [[0 for _ in range(image.height)] for _ in range(image.width)]
max_magnitude = 0

for x in range(1, image.width - 1):
    for y in range(1, image.height - 1):
        hor_kernel_result = image.getpixel((x - 1, y - 1))[0] * -1  # top-left pixel
        hor_kernel_result += image.getpixel((x + 1, y - 1))[0]  # top-right pixel
        hor_kernel_result += image.getpixel((x - 1, y))[0] * -2  # left pixel
        hor_kernel_result += image.getpixel((x + 1, y))[0] * 2  # right pixel
        hor_kernel_result += image.getpixel((x - 1, y + 1))[0] * -1  # bottom-left pixel
        hor_kernel_result += image.getpixel((x + 1, y + 1))[0]  # bottom-right pixel

        ver_kernel_result = image.getpixel((x - 1, y - 1))[0]  # top-left pixel
        ver_kernel_result += image.getpixel((x, y - 1))[0] * 2  # top pixel
        ver_kernel_result += image.getpixel((x + 1, y - 1))[0]  # top-right pixel
        ver_kernel_result += image.getpixel((x - 1, y + 1))[0] * -1 # bottom-left pixel
        ver_kernel_result += image.getpixel((x, y + 1))[0] * -2 # bottom pixel
        ver_kernel_result += image.getpixel((x + 1, y + 1))[0] * -1 # bottom-right pixel

        magnitude = math.sqrt(hor_kernel_result**2 + ver_kernel_result**2)
        energy_map[x][y] = magnitude
        max_magnitude = max(max_magnitude, magnitude)

energy_image = Image.new("RGB", (image.width, image.height))

for x in range(image.width):
    for y in range(image.height):
        color_component = int((energy_map[x][y] * 255) // max_magnitude)
        energy_image.putpixel((x, y), (color_component, color_component, color_component))

energy_image.save("energy.jpg")
