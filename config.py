# Parameters for the program
import os
dir = os.path.dirname(os.path.abspath(__file__))

ENCODED_IMAGES_PATH = os.path.join(dir, "FinetunedEncoder/encoded_images_numpy")
MODEL_PATH = os.path.join(dir, "FinetunedEncoder/encoder.onnx")

#Pretrained
PRETRAINED_ENCODED_IMAGES_PATH = os.path.join(dir, "PretrainedEncoder/encoded_images_pretrained_numpy")
PRETRAINED_MODEL_PATH = os.path.join(dir, "PretrainedEncoder/encoder_pretrained.onnx")

#Indexing for each encoder
INDEXING_DIR = "Indexes"
MATCHING_FILE = os.path.join("Indexes", "matching.csv")

# Settings json
SETTINGS_PATH = os.path.join(dir, "settings.json") 
