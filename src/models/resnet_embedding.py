import os

import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

from env_vars import IMG_SIZE


def get_model():
    # ---------- LOAD RESNET50 BASE ----------
    resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    feature_extractor = Model(inputs=resnet_base.input, outputs=resnet_base.output)

    return feature_extractor


def load_and_preprocess_images(dataset_path, max_images=50):
    embeddings = []
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][
                  :max_images]

    for img_name in image_files:
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img)
        embeddings.append(img)

    if not embeddings:
        return None

    input_batch = np.stack(embeddings)
    feature_extractor = get_model()
    features = feature_extractor.predict(input_batch, verbose=0)
    return np.mean(features, axis=0)  # Mean embedding per dataset
