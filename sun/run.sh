#!/bin/bash
python label_to_mask.py
python data_processing.py
python train.py --model unet