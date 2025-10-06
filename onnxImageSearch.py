import numpy as np
import onnxruntime as ort
from joblib import load

from config import ENCODED_IMAGES_PATH, IMAGE_MODEL_PATH
from distanceFunctions import (chebyshev_dist, cos_dist, euclidean_dist,
                               jaccard_dist, manhattan_dist)
from VLMEncoder.mobileclip_image_transforms import load_image


def main():
    """Prepare the encoded images file and test a search"""
    import os

    # Parameters
    model_path = IMAGE_MODEL_PATH
    images_dir = "images"
    images_names = os.listdir(images_dir)
    images_index = [image_name.split(".")[0] for image_name in images_names]
    images_paths = [os.path.join(images_dir, image_name)
                    for image_name in images_names]
    image_path = "images/209230.jpg"
    encoded_images_path = ENCODED_IMAGES_PATH
    # encoding_dict = load("siamese_model/encoded_images")

    # Onnx operations: use model on image
    session = load_onnx_model(model_path)
    image = load_image(image_path)
    input_image_encoding = encode(session, image)

    # encode whole dataset
    encoding_dict = {}
    if not os.path.exists(encoded_images_path):
        encode_whole_dataset(
            encoding_dict, session, encoded_images_path, images_index, images_paths
        )
    else:
        encoding_dict = load(encoded_images_path)

    # Search best matches
    n_images = len(encoding_dict.keys())
    images_index = list(encoding_dict.keys())
    cos_values = np.zeros(n_images)
    for i, image_encoding in enumerate(encoding_dict.values()):
        cos_values[i] = cos_dist(input_image_encoding, image_encoding)
    # [::-1] is used to invert the sort if we use a similarity instead of a distance
    # sorted_results = np.argsort(cos_values)[::-1]
    sorted_results = np.argsort(cos_values)
    sorted_indexes = [int(images_index[result]) for result in sorted_results]
    print(sorted_indexes[:10])


# https://www.codeproject.com/Articles/5278507/Using-Portable-ONNX-AI-Models-in-Python
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session


# https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb
def normalize(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype("float32")

    # normalize
    mean_vec = np.expand_dims(np.array([0.485, 0.456, 0.406]), (1, 2))
    stddev_vec = np.expand_dims(np.array([0.229, 0.224, 0.225]), (1, 2))
    norm_img_data = (img_data / 255 - mean_vec) / stddev_vec

    # add batch channel
    norm_img_data = np.expand_dims(norm_img_data, 0).astype("float32")
    return norm_img_data


def encode(session, image):
    image = normalize(image)
    input_name = session.get_inputs()[0].name
    encoding = session.run(None, {input_name: image})
    return encoding


def encode_whole_dataset(
    encoding_dict, session, encoded_images_path, images_index, images_paths, signal=None
):
    # Encode the image database with the trained encoder
    from joblib import dump

    for image_index, image_path in zip(images_index, images_paths):
        # Skip if the file cannot be loaded as an image
        try:
            image = load_image(image_path)
        except:
            continue
        image_encoding = encode(session, image)
        encoding_dict[image_index] = image_encoding
        if signal != None:
            signal.emit()
        else:
            print(image_index, "/", len(images_paths), end="\r")
    dump(encoding_dict, encoded_images_path)
    return encoding_dict


def compare(encoding1, encoding2, dist):
    return dist(encoding_1, encoding_2)


if __name__ == "__main__":
    main()
