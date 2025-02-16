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
    reload_segmodel(GeoTr_Seg_model.msk, 'D:/NIK_CINICHKA/cb/NN/Doc/model_pretrained/geotr.pth')
    reload_model(GeoTr_Seg_model.GeoTr, 'D:/NIK_CINICHKA/cb/NN/Doc/model_pretrained/geotr.pth')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process document image')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save processed image')
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} not found")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    process_and_save_image(args.input, args.output)