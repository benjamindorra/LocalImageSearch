import onnxruntime as ort
import numpy as np
from joblib import load
from PIL import Image
from config import MODEL_PATH, ENCODED_IMAGES_PATH

def main():
    """Prepare the encoded images file and test a search"""
    import os
    # Parameters
    model_path = MODEL_PATH
    images_dir = "images"
    images_names = os.listdir(images_dir)
    images_index = [image_name.split(".")[0] for image_name in images_names]
    images_paths = [os.path.join(images_dir, image_name)
            for image_name in images_names]
    image_path = "images/209230.jpg"
    encoded_images_path = ENCODED_IMAGES_PATH
    #encoding_dict = load("siamese_model/encoded_images")

    # Onnx operations: use model on image
    session = load_onnx_model(model_path)
    image = load_image(image_path)
    input_image_encoding = encode(session,image) 
    
    #encode whole dataset
    encoding_dict = {}
    if not os.path.exists(encoded_images_path):
        encode_whole_dataset(encoding_dict, session, encoded_images_path,
                images_index, images_paths)
    else:
        encoding_dict = load(encoded_images_path)

    # Search best matches
    n_images = len(encoding_dict.keys())
    images_index = list(encoding_dict.keys())
    cos_values = np.zeros(n_images)
    for i, image_encoding in enumerate(encoding_dict.values()):
        cos_values[i] = cos_sim(input_image_encoding, image_encoding)
    sorted_results = np.argsort(cos_values)[::-1]
    sorted_indexes = [int(images_index[result])
            for result in sorted_results]
    print(sorted_indexes[:10])

#https://www.codeproject.com/Articles/5278507/Using-Portable-ONNX-AI-Models-in-Python
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

#https://www.tutorialexample.com/best-practice-to-calculate-cosine-distance-between-two-vectors-in-numpy-numpy-tutorial/
def cos_sim(vector1, vector2):
    vector1 = np.squeeze(vector1)
    vector2 = np.squeeze(vector2)
    v1_norm = np.linalg.norm(vector1)
    v2_norm = np.linalg.norm(vector2)
    prod = np.dot(vector1, vector2)
    epsilon = 1e-8
    cos = prod/(v1_norm*v2_norm+epsilon)
    return cos 


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224,224))
    # Remove alpha channel if there is one
    image = image.convert('RGB')
    return np.array(image).transpose(2,0,1)


#https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb
def normalize(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.expand_dims(np.array([0.485, 0.456, 0.406]),(1,2))
    stddev_vec = np.expand_dims(np.array([0.229, 0.224, 0.225]), (1,2))
    norm_img_data = (img_data/255 - mean_vec) / stddev_vec

    #add batch channel
    norm_img_data = np.expand_dims(norm_img_data, 0).astype('float32')
    return norm_img_data


def encode(session, image):
    image = normalize(image)
    input_name = session.get_inputs()[0].name
    encoding = session.run(None, {input_name: image})
    return encoding


def encode_whole_dataset(encoding_dict, session, encoded_images_path,
        images_index, images_paths, signal=None):
    # Encode the image database with the trained encoder
    from joblib import dump
    for image_index, image_path in zip(images_index, images_paths):
      # Skip if the file cannot be loaded as an image
      try:
        image = load_image(image_path)
      except:
        continue
      image_encoding = encode(session,image)
      encoding_dict[image_index] = image_encoding
      if signal != None:
        signal.emit()
      else:
        print(image_index,'/',len(images_paths),end='\r')
    dump(encoding_dict, encoded_images_path)
    return encoding_dict


def compare(encoding1, encoding2):
    return cos_sim(encoding_1,encoding_2)


if __name__=="__main__":
        main()
