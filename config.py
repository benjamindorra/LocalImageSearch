# Parameters for the program
import os
dir = os.path.dirname(os.path.abspath(__file__))

ENCODED_IMAGES_PATH = os.path.join(dir, "VLMEncoder/encoded_images_numpy")
IMAGE_MODEL_PATH = os.path.join(dir, "VLMEncoder/mobileclip2_s0_image_encoder.onnx")
TEXT_MODEL_PATH = os.path.join(dir, "VLMEncoder/mobileclip2_s0_text_encoder.onnx")

#Indexing for each encoder
INDEXING_DIR = "Indexes"
MATCHING_FILE = os.path.join("Indexes", "matching.csv")

# Settings json
SETTINGS_PATH = os.path.join(dir, "settings.json")
