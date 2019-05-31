# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 05:39:50 2019

@author: yiyuezhuo
"""

import numpy as np
from PIL import ImageFont, ImageDraw, Image
fontpath = "SIMYOU.TTF"
font = ImageFont.truetype(fontpath, 25) # 20 is font size

def draw_chinese(img_processed, text, pos, fill = (255, 0, 0)):
    img_pil = Image.fromarray(img_processed)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font = font, fill = fill)
    return np.array(img_pil)