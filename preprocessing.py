import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def load_preprocess_images(csv_path):
    # load CSV into DataFrame
    df = pd.read_csv(csv_path)

    # filter for 40X magnification
    df_filtered = df[df['mag'] == 40]

    # prepare image and label lists
    images = []
    labels = []

    for _, row in df_filtered.iterrows():
        try:
            # load and resize image
            img = Image.open(row['filename']).convert('RGB')
            img_resized = img.resize((224, 224))

            # normalize pixel values and flatten
            img_array = np.array(img_resized) / 255.0
            img_flattened = img_array.flatten()

            images.append(img_flattened)
            labels.append(row['tumor_class'])

        except Exception as e:
            print(f"Error processing file {row['filename']}: {e}")

    # convert to NumPy arrays
    X = np.array(images)
    y = np.array(labels)

    return X, y


def split_and_apply_pca(X, y, n_components=200, random_state=42):
    # first split into train 60% and temp 40%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )

    # then split temp into val 20% and test 20%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    # fit PCA on training set
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)

    # apply PCA transformation to val and test sets
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca
