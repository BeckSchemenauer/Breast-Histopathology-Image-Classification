import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def load_preprocess_images(csv_path, filter_40x=False, flatten=False):
    # load CSV into DataFrame
    df = pd.read_csv(csv_path)

    if filter_40x:
        # filter for 40X magnification
        df = df[df['mag'] == 40]

    # prepare image, label, and ID lists
    images = []
    labels = []
    ids = []

    for _, row in df.iterrows():
        try:
            # load and resize image
            img = Image.open(row['filename']).convert('RGB')
            img_resized = img.resize((224, 224))

            # normalize pixel values
            img_array = np.array(img_resized) / 255.0
            if flatten:
                img_array = img_array.flatten()

            images.append(img_array)
            labels.append(row['tumor_class'])
            ids.append(row['patient_id'])

        except Exception as e:
            print(f"Error processing file {row['filename']}: {e}")

    return np.array(images), np.array(labels), np.array(ids)


def split_data(X, y, random_state=42):
    # first split into train 60% and temp 40%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )

    # then split temp into val 20% and test 20%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_pca(X_train, X_val, X_test, y_train, y_val, y_test, n_components=200):
    # fit PCA on training set
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)

    # apply PCA transformation to val and test sets
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca
