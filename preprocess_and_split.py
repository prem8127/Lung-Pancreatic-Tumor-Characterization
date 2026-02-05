import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

IMG_SIZE = 224
DATASET_PATH = "data/lung"
CLASSES = ["normal", "benign", "malignant"]

def load_images(dataset_path):
    images = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        class_path = os.path.join(dataset_path, class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                images.append(img)
                labels.append(label)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)
    return images, labels

# Load data
X, y = load_images(DATASET_PATH)

print("Total images:", X.shape[0])
print("Class distribution:", Counter(y))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Save arrays (optional but useful)
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Data preprocessing & split completed successfully ✅")
