
# coding: utf-8

# In[3]:


import cv2
from PIL import Image
import numpy as np
from skimage import io
from path import Path
from utils.en_de import Encode_Decode

en_de_tool = Encode_Decode()
dataset_path = ['train_set/','val_set/']
for item in dataset_path:
    file_path = [i for i in Path(f'{item}/').files() if 'label' in i.name]
    for i in file_path:
        temp = io.imread(i)
        temp = en_de_tool.encode_segmap(temp)
        cv2.imwrite( f'{item}' + i.stem + '_mask.png',temp)

