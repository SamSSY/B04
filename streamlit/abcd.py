from PIL import Image
from test import runfrist

img_path = "new3_2.jpg"  # 图像文件路径
img = Image.open(img_path)
print(runfrist(img_path))
