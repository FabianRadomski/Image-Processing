folder = 'SameSizeNumbers'

from os import walk
_, _, filenames = next(walk(folder))

from PIL import Image


for img_path in filenames:
    im = Image.open(folder + '/' + img_path)
    max_y = 0
    max_x = 0
    for x in range(im.width):
            for y in range(im.height):
                if im.getpixel((x, y)) != 0:
                    max_y = max(max_y, y)
                    max_x = max(max_x, x)
    im.crop((0, 0, max_x + 1, max_y + 1)).save(folder + '/' + img_path)