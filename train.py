import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

data = []
labels = []

def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')
    return features

for category in ["real", "fake"]:
    path = os.path.join("dataset", category)
    label = 0 if category == "real" else 1

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            features = extract_features(img_path)
            data.append(features)
            labels.append(label)
        except:
            pass

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC(probability=True)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("Accuracy:", acc)

joblib.dump(model, "model.pkl")
print("Model saved!")



