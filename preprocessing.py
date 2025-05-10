import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit


def load_preprocess_images(csv_path):
    # Load CSV into DataFrame
    df = pd.read_csv(csv_path)

    # filter for 40X magnification
    df_filtered = df[df['mag'] == 40]

    # prepare image and label lists
    images = []
    labels = []
    patient_ids = []

    for _, row in df_filtered.iterrows():
        try:
            # Load and resize image
            img = Image.open(row['filename']).convert('RGB')
            img_resized = img.resize((224, 224))

            # normalize and flatten
            img_array = np.array(img_resized) / 255.0
            images.append(img_array.flatten())

            # collect label and patient id
            labels.append(row['tumor_class'])
            patient_ids.append(row['patient_id'])

        except Exception as e:
            print(f"error processing {row['filename']}: {e}")

    X = np.stack(images)
    y = np.array(labels)
    groups = np.array(patient_ids)

    return X, y, groups


def split_and_apply_pca(X, y, groups, n_components=200, random_state=42):
    # first, split out 60% train vs 40% temp by patient_id
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(X, y, groups))
    X_train, y_train = X[train_idx], y[train_idx]
    groups_temp = groups[temp_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]

    # then split temp 50/50 → val and test, again by patient_id
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_rel_idx, test_rel_idx = next(gss2.split(X_temp, y_temp, groups_temp))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # fit PCA on train only
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)

    # apply PCA to val and test
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca
