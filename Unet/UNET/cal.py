import cv2
import os
from glob import glob

def count_white_pixels(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    white_pixels = cv2.countNonZero(img)
    width, height = img.shape
    total_pixels = width * height
    return white_pixels, total_pixels

if __name__ == "__main__":
    path = os.path.join("..","new_data","test","mask")
    file_path = os.path.join(path,"result.txt")  # File path
    img_total = sorted(glob(os.path.join(path, "*.jpg")))

    with open(file_path, "w") as file:

        file.write("相片\t百分比\t像素數量\n")  # Write file header

        for image_path in img_total:
            name = os.path.splitext(os.path.basename(image_path))[0]
            white_pixels, total_pixels = count_white_pixels(image_path)
            white_ratio = (white_pixels / total_pixels) * 100

            result_str = "{: <10}{: <9}{: <10}".format(name, '%.2f' % white_ratio + "%", white_pixels)
            print(result_str)

            file.write(result_str + "\n")  # Write results to file
