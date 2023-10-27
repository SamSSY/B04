import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "train", "image", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "mask", "*.jpg")))

    test_x = sorted(glob(os.path.join(path, "test", "image", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "test", "mask", "*.jpg")))

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (image_path, mask_path) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = os.path.splitext(os.path.basename(image_path))[0]
        #print(image_path)
        #print(name)
        
        # Read image and mask
        x = cv2.imread(image_path, cv2.IMREAD_COLOR)
        y = imageio.mimread(mask_path)[0]
        #print(x.shape)
        
        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]
            
            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.jpg"
            tmp_mask_name = f"{name}_{index}.jpg"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

        

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
   
    data_path = os.path.join(".", "date")
    #print(data_path)
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir(os.path.join(".", "new_data","train","image"))
    create_dir(os.path.join(".", "new_data","train","mask"))
    create_dir(os.path.join(".", "new_data","test","image"))
    create_dir(os.path.join(".", "new_data","test","mask"))

    """ Data augmentation """
    augment_data(train_x, train_y, os.path.join(".", "new_data/train/"), augment=True)
    augment_data(test_x, test_y, os.path.join(".", "new_data/test/"), augment=False)

    new_data_path = os.path.join(".", "new_data")
    #print(new_data_path)
    (new_train_x, new_train_y), (new_test_x, new_test_y) = load_data(new_data_path)
    
    print(f"Train: {len(new_train_x)} - {len(new_train_y)}")
    print(f"Test: {len(new_test_x)} - {len(new_test_y)}")
  
