import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')

def load_and_preprocess_images(dataframe):
    images = []
    labels = []
    for idx, row in dataframe.iterrows():
        img_path = row['image:FILE']
        label = row['category']
        image = Image.open(img_path).convert("RGB").resize((128, 128))
        images.append(np.array(image).flatten())  
        labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = load_and_preprocess_images(train_data)
X_val, y_val = load_and_preprocess_images(val_data)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

svm_model = SVC(kernel='linear', C=1)  
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_val)
print(f"SVM Accuracy: {accuracy_score(y_val, y_pred):.2f}")
print(classification_report(y_val, y_pred))
