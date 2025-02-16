import cv2
import pytesseract
import csv
import os
import re
from seg import U2NETP
from GeoTr import GeoTr
from IllTr import IllTr
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from glob import glob
import argparse
import warnings

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
warnings.filterwarnings('ignore')

class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)

    def forward(self, x):
        msk, _1, _2, _3, _4, _5, _6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x
        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        return bm

def reload_model(model, path=""):
    if path:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def reload_segmodel(model, path=""):
    if path:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def process_and_save_image(input_path, output_path):
    GeoTr_Seg_model = GeoTr_Seg()
    reload_segmodel(GeoTr_Seg_model.msk, r'D:/NIK_CINICHKA/cb/NN/Doc/model_pretrained/geotr.pth')
    reload_model(GeoTr_Seg_model.GeoTr, r'D:/NIK_CINICHKA/cb/NN/Doc/model_pretrained/geotr.pth')
    GeoTr_Seg_model.eval()

    input_image = Image.open(input_path).convert('RGB')
    im_ori = np.array(input_image)[:, :, :3] / 255.
    h, w, _ = im_ori.shape

    im = cv2.resize(im_ori, (288, 288))
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0)

    with torch.no_grad():
        bm = GeoTr_Seg_model(im)
        bm = bm.cpu()
        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)

        out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
        img_geo = ((out[0] * 255).permute(1, 2, 0).numpy()).astype(np.uint8)

    result_image = Image.fromarray(img_geo)
    result_image.save(output_path)
    print(f"Image successfully saved to {output_path}")

def clean_text(text):
    cleaned = re.sub(r'["\'\\]', '', text)
    return cleaned

def cv2_rebuild(image_path, output_dir=r"D:\NIK_CINICHKA\cb\NN\Doc\proc_after_mn"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem
    filename = f"processed_{base_name}.jpg"

    output_path = os.path.join(r"D:\NIK_CINICHKA\cb\NN\Doc\proc_after_mn", filename)
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
    
    filename = f"{Path(image_path).name}"
    processed_path = os.path.join(output_dir, filename)
    cv2.imwrite(processed_path, binary)
    print(processed_path)
    print(process_and_save_image(processed_path, output_path))
    return processed_path

def process_images(input_pattern, output_csv="result.csv"):
    image_paths = glob(input_pattern)
    
    if not image_paths:
        raise ValueError("Файлы не найдены по указанному пути")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        for idx, image_path in enumerate(image_paths, 1):
            try:
                processed_path = cv2_rebuild(image_path)
                
                print(f"Обработано: {idx}/{len(image_paths)}")
                
            except Exception as e:
                error_msg = f"Ошибка в {Path(image_path).name}: {str(e)}"
                print(error_msg)
                writer.writerow([f'{Path(image_path).name}:"{clean_text(error_msg)}"'])

if __name__ == "__main__":
    process_images(
        input_pattern=r"D:\NIK_CINICHKA\cb\NN\Doc\proc_after_nn\processed_file_*.jpg", 
        output_csv="text_results.csv"
    )