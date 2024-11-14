import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
        image = Image.open(img_path).convert("RGB").resize((400, 400))  # Resize to 32x32 to reduce complexity
        images.append(np.array(image).flatten())  # Flatten the image
        labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = load_and_preprocess_images(train_data)
X_val, y_val = load_and_preprocess_images(val_data)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_val)
print(f"Naive Bayes Accuracy: {accuracy_score(y_val, y_pred):.2f}")
print(classification_report(y_val, y_pred))
