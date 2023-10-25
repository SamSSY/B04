
import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import imageio
import torch
import sys
import getopt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

def count_white_pixels(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    white_pixels = cv2.countNonZero(img)
    width, height = img.shape
    total_pixels = width * height
    return white_pixels, total_pixels

def runfrist(in_image):
    
    """ Seeding """
    seeding(42)

    """ Load dataset """
    #test_x = Image.open(in_image)
    #test_x = in_image
    #test_x = sorted(glob(os.path.join("..",s_path,"*")))
    #test_y = sorted(glob("../new_data/test/mask/*"))#

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = os.path.join(".","files","20230718_1257_checkpoint.pth")
    #checkpoint_path = os.path.join("..",w_path)#############################

    """ Load the checkpoint """
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    #metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    #time_taken = []

    #for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
    #for i, x in tqdm(enumerate(test_x), total=len(test_x)):
        
    """ Extract the name """
    #name = x.split("/")[-1].split(".")[0]
    #name = os.path.splitext(os.path.basename(test_x))[0]

    """ Reading image """
    image = cv2.imread(in_image, cv2.IMREAD_COLOR) ## (512, 512, 3)
    ## image = cv2.resize(image, size)
    x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
    x = x/255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)

    with torch.no_grad():
        """ Prediction and Calculating FPS """
        start_time = time.time()
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        total_time = time.time() - start_time
        #time_taken.append(total_time)

        #score = calculate_metrics(y, pred_y)
        #metrics_score = list(map(add, metrics_score, score))
        pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
        pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

    """ Saving masks """
    #ori_mask = mask_parse(mask)
    pred_y = mask_parse(pred_y)
    line = np.ones((size[1], 10, 3)) * 128

    cv2.imwrite("cach.jpg", pred_y * 255)
    white_pixels, total_pixels = count_white_pixels("cach.jpg")
    white_ratio = (white_pixels / total_pixels) * 100
    os.remove("cach.jpg")
    
    return white_ratio


def run(image_file):
    """ Seeding """
    seeding(42)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = os.path.join(".","files","20230718_1257_checkpoint.pth")
    
    """ Load the checkpoint """
    device = torch.device('cpu')  # Use CPU for inference

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Reading image """
    # Read the uploaded image using OpenCV
    image_data = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), -1)

    # Ensure image_data has the expected shape (512, 512, 3)
    if image_data.shape != (512, 512, 3):
        raise ValueError("Input image_data should have shape (512, 512, 3)")

    # Normalize the image data
    x = image_data / 255.0  # Normalize the image data
    x = np.transpose(x, (2, 0, 1))  # (3, 512, 512)
    x = np.expand_dims(x, axis=0)  # Add batch dimension (1, 3, 512, 512)
    x = x.astype(np.float32)

    # Convert x to a PyTorch tensor
    x = torch.from_numpy(x).to(device)

    with torch.no_grad():
        # Prediction and Calculating FPS
        start_time = time.time()
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        total_time = time.time() - start_time
        pred_y = pred_y[0].cpu().numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

    # Calculate the white pixel ratio
    white_pixels = np.sum(pred_y == 1)
    total_pixels = pred_y.size
    white_ratio = (white_pixels / total_pixels) * 100

    return white_ratio
