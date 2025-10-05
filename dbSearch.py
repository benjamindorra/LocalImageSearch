"""
Program for keywords and image searching in a set of images

Backend for the GUI (python dbSearchGui.py)
"""

from joblib import load
import os
import sys
import numpy as np
import pandas as pd
import unicodedata
import onnxImageSearch as imsearch
from VLMEncoder.openclip_tokenizer import encode_text
from VLMEncoder.mobileclip_image_transforms import load_image, encode_image
from distanceFunctions import cos_dist, euclidean_dist, manhattan_dist, chebyshev_dist, jaccard_dist
dist_dict = {'Cosine':cos_dist, 'Euclidean':euclidean_dist, 'Manhattan':manhattan_dist, 'Chebyshev':chebyshev_dist, 'Jaccard':jaccard_dist}

def get_imgs(imgDir):
  #imgDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
  imgNames = os.listdir(imgDir)
  imgPaths = [os.path.join(imgDir,imgName) for imgName in imgNames]
  return imgDir,imgNames,imgPaths

def formatResults(result,imgDir):
  result_list = []
  for row in result.iterrows():
      row_infos = []
      row_infos.append(row[1].to_string() + "\nIndex "+str(row[0])+"\n")
      row_infos.append(os.path.join(imgDir,str(row[1]["name"])))
      result_list.append(row_infos)

  return result_list

def searchImg(input_image_path,df,model_path,encoded_images_path,dist,signal=None,tid=None):
  # Select distance function for the similarity search
  dist_func = dist_dict[dist]
  # Load all images encoding
  encoding_dict = load(encoded_images_path)
  
  # Load encoder
  session = imsearch.load_onnx_model(model_path)

  # Load and encode input image
  input_image = imsearch.load_image(input_image_path)
  input_image_encoding = imsearch.encode(session, input_image)

  # Get similarity to the whole dataset
  n_images = len(encoding_dict.keys())
  images_index = list(encoding_dict.keys())
  dist_values = np.zeros(n_images)
  for i, image_encoding in enumerate(encoding_dict.values()):
    dist_values[i] = dist_func(input_image_encoding, image_encoding)
    if signal != None:
        signal.emit(tid)
    else:
        print(i,'/',nImages,end='\r')

  # Sort dataframe by decreasing order of similarity
  returnValue = []
  title = 'Results for image '+str(input_image_path)
  sorted_results = np.argsort(dist_values)
  result = df.reindex(sorted_results)
  result = result.reset_index(drop=True)
  returnValue.append([title, result])

  return returnValue

def searchText(input_text,df,model_path,encoded_images_path,dist,signal=None,tid=None):
  # Select distance function for the similarity search
  dist_func = dist_dict[dist]
  # Load all images encoding
  encoding_dict = load(encoded_images_path)

  # Load encoder
  session = imsearch.load_onnx_model(model_path)

  # Encode input text
  input_text_encoding = encode_text(session, input_text)

  # Get similarity to the whole dataset
  n_images = len(encoding_dict.keys())
  images_index = list(encoding_dict.keys())
  dist_values = np.zeros(n_images)
  for i, image_encoding in enumerate(encoding_dict.values()):
    dist_values[i] = dist_func(input_text_encoding, image_encoding)
    if signal != None:
        signal.emit(tid)
    else:
        print(i,'/',nImages,end='\r')

  # Sort dataframe by decreasing order of similarity
  returnValue = []
  title = 'Results for query '+str(input_text)
  sorted_results = np.argsort(dist_values)
  result = df.reindex(sorted_results)
  result = result.reset_index(drop=True)
  returnValue.append([title, result])

  return returnValue


def getFromTo(values, imgDir, start, end):
  """Select results from a search"""
  resultTitle, resultDf = values[0]
  resultDf = resultDf.iloc[start:end]
  returnValues = []
  returnValues.append([resultTitle, formatResults(resultDf, imgDir)])
  return returnValues


def getNumChunks(values, chunkSize):
  """Get number of chunck of length chunkSize in the dataframe"""
  resultsDf = values[0][1]
  if type(resultsDf) == list:
    # No results
    numChunks = 0
  else:
    numChunks = resultsDf.index.size // chunkSize + 1
  return numChunks
  
