"""
MIT License

Copyright © 2024 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import onnxruntime as ort
from mobileclip_image_transforms import encode, load_image
from openclip_tokenizer import SimpleTokenizer
from scipy.special import softmax

image = load_image("c1bd6e30409ac7208a62a8acc1b9d466.jpg")
tokenizer = SimpleTokenizer()
raw_text = ["lizard", "pigeon", "cat", "penguin"]
# raw_text = ["penguin"]
texts = [tokenizer(w) for w in raw_text]

# text_model_path = "mobileclip2_s0_text_encoder.onnx"
# image_model_path = "mobileclip2_s0_image_encoder.onnx"
text_model_path = "mobileclip2_s0_text_encoder.onnx"
image_model_path = "mobileclip2_s0_image_encoder.onnx"


# https://www.codeproject.com/Articles/5278507/Using-Portable-ONNX-AI-Models-in-Python
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session


text_session = load_onnx_model(text_model_path)
image_session = load_onnx_model(image_model_path)

image_encoding = encode(image_session, image)
input_name = text_session.get_inputs()[0].name
text_encoding = []
for text in texts:
    text_encoding.append(text_session.run(None, {input_name: text}))
text_encoding = np.concatenate(text_encoding, axis=1)


image_encoding /= np.linalg.norm(image_encoding, axis=-1, keepdims=True)
text_encoding /= np.linalg.norm(text_encoding, axis=-1, keepdims=True)
image_encoding = np.squeeze(image_encoding, axis=0)
text_encoding = np.squeeze(text_encoding, axis=0)
text_probs = softmax((100.0 * image_encoding @ text_encoding.T), axis=-1)

print("Label probs:", list(
    zip(raw_text, np.split(text_probs, len(raw_text), axis=-1))))
