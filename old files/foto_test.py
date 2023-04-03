import os
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm


""" Load the test images """
test_images = glob("/home/oguzay/Documents/GitHub/Lane-Segmentation/Lane-Segmentation/veri/**.png")

""" Load the model """
model = tf.keras.models.load_model("/home/oguzay/Documents/GitHub/Lane-Segmentation/Lane-Segmentation/weights/unet_12_03_2023_6.h5",compile=False)

for path in tqdm(test_images, total=len(test_images)):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    original_image = x
    h, w, _ = x.shape

    x = cv2.resize(x, (512, 512))
    x = np.expand_dims(x,axis=-1)
    x = x/255.0
    x = x.astype(np.float32)

    x = np.expand_dims(x, axis=0)
    pred_mask = model.predict(x)[0]

    pred_mask = np.concatenate(
        [
            pred_mask,
            pred_mask,
            pred_mask
        ], axis=2)
    pred_mask = (pred_mask > 0.5) * 255
    pred_mask = pred_mask.astype(np.float32)
    pred_mask = cv2.resize(pred_mask, (w, h))

    original_image = original_image.astype(np.float32)

    alpha = 1.0
    output = cv2.addWeighted(pred_mask, alpha, original_image, 1-alpha, 0, original_image)
    output = cv2.resize(output, (1280,720))
    cv2.imshow("dsdasa", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    name = path.split("/")[-1]
    cv2.imwrite(f"save_images/{name}", original_image)
    """