import copy
import os

import cv2
import numpy as np
import requests


def run_skyseg(onnx_session, input_size, image):
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    if max_value > min_value:
        onnx_result = (onnx_result - min_value) / (max_value - min_value)
    else:
        onnx_result = np.zeros_like(onnx_result)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")
    return onnx_result


def segment_sky(image_path, onnx_session, mask_filename=None):
    if mask_filename is None:
        raise ValueError("mask_filename must not be None")
    image = cv2.imread(image_path)
    if image is None:
        return None

    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def download_file_from_url(url, filename):
    try:
        response = requests.get(url, allow_redirects=False, timeout=30)
        response.raise_for_status()
        if response.status_code == 302:
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True, timeout=120)
            response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as exc:
        print(f"Error downloading file: {exc}")
