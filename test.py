import cv2
import pytesseract
import csv
import os
import re
from pathlib import Path
from glob import glob
from seg import U2NETP
from GeoTr import GeoTr
from IllTr import IllTr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
import argparse
import warnings

warnings.filterwarnings('ignore')


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def process_image(image_path, output_path):
  try:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (2480, 3508))
    if image is None:
        raise ValueError(f"Не удалось загрузить: {image_path}")
    

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.bilateralFilter(gray, d=10, sigmaColor=10, sigmaSpace=75)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
      
    processed_image_path = "output/processed_image.jpg"
    cv2.imwrite(processed_image_path, gray)

    text = pytesseract.image_to_string(binary, lang='rus')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Обработка изображения завершена. Результат сохранен в: {output_path}")

  except Exception as e:
      print(f"Произошла ошибка при обработке изображения: {e}")


image_folder = 'proc_after_mn'
output_folder = 'output2'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(image_folder):
  if filename.endswith(('.jpg', '.jpeg', '.png')):
    image_path = os.path.join(image_folder, filename)
    output_path = os.path.join(output_folder, filename[:-4] + '.txt')
    process_image(image_path, output_path)
