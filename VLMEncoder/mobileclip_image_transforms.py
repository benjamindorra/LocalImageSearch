"""
Preprocessing before feeding the images to mobileclip image encoder
transforms': [Resize(size=256, interpolation=bicubic, max_size=None, antialias=True), CenterCrop(size=(256, 256)), <function _convert_to_rgb at 0x7ff5c44918a0>, ToTensor(), Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]}
"""

import numpy as np
from PIL import Image


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    # Remove alpha channel if there is one
    image = image.convert("RGB")
    return np.array(image).transpose(2, 0, 1)


def normalize(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype("float32")

    # normalize
    mean_vec = np.expand_dims(np.array([0.48145466, 0.4578275, 0.40821073]), (1, 2))
    stddev_vec = np.expand_dims(np.array([0.26862954, 0.26130258, 0.27577711]), (1, 2))
    norm_img_data = (img_data / 255 - mean_vec) / stddev_vec

    # add batch channel
    norm_img_data = np.expand_dims(norm_img_data, 0).astype("float32")
    return norm_img_data


def encode_image(session, image):
    image = normalize(image)
    input_name = session.get_inputs()[0].name
    encoding = session.run(None, {input_name: image})
    return encoding
