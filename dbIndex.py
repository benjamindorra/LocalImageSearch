"""
Index or reindex an image directory
"""

import os

import pandas as pd
from PIL import Image

import config
from onnxImageSearch import encode_whole_dataset, load_onnx_model


def create_df(images_dir, dest):
    """
    Create an infos dataframe of the matching directory
    For now: only image names
    """
    image_names = []
    names = os.listdir(images_dir)
    for name in names:
        # Skip if the file cannot be loaded as an image
        try:
            path = os.path.join(images_dir, name)
            image = Image.open(path)
        except:
            continue
        image_names.append(name)
    df = pd.DataFrame()
    df["name"] = image_names
    df.to_csv(dest, index=False)


def create_embeddings(images_dir, dest, encoder_path, signal=None):
    """Create embeddings for the selected images directory"""
    images_names = os.listdir(images_dir)
    images_paths = [os.path.join(images_dir, image_name) for image_name in images_names]
    session = load_onnx_model(encoder_path)
    encode_whole_dataset({}, session, dest, images_names, images_paths, signal)


def indexDir(images_dir, encoder_path, signal=None):
    """Create embeddings, metadata dataframe, and save paths in the matching csv"""
    dir_dest = os.path.join(
        os.path.dirname(encoder_path), config.INDEXING_DIR, os.path.basename(images_dir)
    )
    if not os.path.exists(dir_dest):
        os.mkdir(dir_dest)
    dest_embeddings = os.path.join(dir_dest, "embeddings.npy")
    dest_df = os.path.join(dir_dest, "df.csv")
    create_embeddings(images_dir, dest_embeddings, encoder_path, signal)
    create_df(images_dir, dest_df)
    csv_path = os.path.join(os.path.dirname(encoder_path), config.MATCHING_FILE)
    if os.path.exists(csv_path):
        matching_df = pd.read_csv(csv_path)
    else:
        matching_df = pd.DataFrame()
    matching_df = pd.concat(
        [
            matching_df,
            pd.DataFrame(
                [
                    {
                        "images": images_dir,
                        "embeddings": dest_embeddings,
                        "infos": dest_df,
                    }
                ]
            ),
        ]
    )
    matching_df.to_csv(csv_path, index=False)
