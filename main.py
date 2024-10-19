import torch
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Use cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define paths for training and validation images and the labels
train_img_dir = r'C:\Users\georg\OneDrive\Desktop\Side Projects\ObjectDetectionHotdog\ObjectDetectorHotdog\train\images'
val_img_dir = r'C:\Users\georg\OneDrive\Desktop\Side Projects\ObjectDetectionHotdog\ObjectDetectorHotdog\valid\images'
train_labels_dir = r'C:\Users\georg\OneDrive\Desktop\Side Projects\ObjectDetectionHotdog\ObjectDetectorHotdog\train\labels'
val_labels_dir = r'C:\Users\georg\OneDrive\Desktop\Side Projects\ObjectDetectionHotdog\ObjectDetectorHotdog\valid\labels'

# Load YOLOv8 model (pre-trained) and move it to the gpu (or cpu if gpu not found)
model = YOLO('yolov8n.pt').to(device)

# prepare datasets in YOLO format (images + txt labels in YOLO format)
def load_dataset(image_dir, label_dir):
    img_paths = list(Path(image_dir).rglob('*.jpg'))
    label_paths = list(Path(label_dir).rglob('*.txt'))
    return img_paths, label_paths

def check_image_label_mismatch(image_dir, label_dir):
    images = {os.path.splitext(f)[0] for f in os.listdir(image_dir)}  # Strip file extensions
    labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}  # Strip file extensions

    extra_labels = labels - images
    extra_images = images - labels
    print(f"Extra Labels (no corresponding images): {extra_labels}")
    print(f"Extra Images (no corresponding labels): {extra_images}")
    
    return extra_labels, extra_images

print("-------------------------------------------------------------------------------")
print("Checking Training Dataset for Mismatches...")
extra_train_labels, extra_train_images = check_image_label_mismatch(train_img_dir, train_labels_dir)

print("Checking Validation Dataset for Mismatches...")
extra_val_labels, extra_val_images = check_image_label_mismatch(val_img_dir, val_labels_dir)
print("-------------------------------------------------------------------------------")


def delete_extra_labels(extra_labels, label_dir):
    for label in extra_labels:
        label_path = os.path.join(label_dir, label + '.txt')  # Label files are .txt
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"Deleted: {label_path}")
        else:
            print(f"File not found: {label_path}")


def delete_extra_imgs(extra_imgs, img_dir):
    for img in extra_imgs:
        img_path = os.path.join(img_dir, img + '.jpg')  # Label files are .txt
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Deleted: {img_path}")
        else:
            print(f"File not found: {img_path}")

delete_extra_labels(extra_val_labels, val_labels_dir)
delete_extra_labels(extra_train_labels, train_labels_dir)
delete_extra_imgs(extra_val_images, val_img_dir)
delete_extra_imgs(extra_train_images, train_img_dir)


# load training and validation datasets
train_imgs, train_labels = load_dataset(train_img_dir, train_labels_dir)
val_imgs, val_labels = load_dataset(val_img_dir, val_labels_dir)

print("Deleted extra train imgs: ", len(train_labels) == len(train_imgs))
print("Deleted extra val imgs: "  , len(val_labels) == len(val_imgs))

# Train the model using the training data
# will use 5 epochs for time and 16 batches with img size 640 for accuracy
results = model.train(data='data.yaml', 
                      epochs=70,               # Try with 70 epochs
                      imgsz=640,               # Or increase for better accuracy
                      batch=16,                # Keep or increase batch size
                      lr0=0.01,                # Good starting point for LR
                      momentum=0.937,          # Momentum
                      weight_decay=0.0005,     # Standard weight decay
                      label_smoothing=0.1,     # Smooths out labels
                      mosaic=1.0,              # Augmentation
                      flipud=0.0,              # No vertical flip
                      fliplr=0.5,              # Horizontal flip on
                      iou=0.7,                 # IoU threshold
                      name = 'hotdog_detector_test1')

# Test on validation set
metrics = model.val(data='data.yaml', 
                      epochs=70,               # Try with 70 epochs
                      imgsz=640,               # Or increase for better accuracy
                      batch=16,                # Keep or increase batch size
                      lr0=0.01,                # Good starting point for LR
                      momentum=0.937,          # Momentum
                      weight_decay=0.0005,     # Standard weight decay
                      label_smoothing=0.1,     # Smooths out labels
                      mosaic=1.0,              # Augmentation
                      flipud=0.0,              # No vertical flip
                      fliplr=0.5,              # Horizontal flip on
                      iou=0.7                 # IoU threshold
                      )
