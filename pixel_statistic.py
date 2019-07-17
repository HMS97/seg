import os
import glob
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
mpl.rcParams["font.sans-serif"] = ["SimHei"]

labellist = glob.glob("./train_mask/*label.npy")
n_classes = 16
# path = "./train_mask/GF2_PMS1__20150212_L1A0000647768-MSS1_label.npy"
# data = np.load(path)
# row,col = data.shape
# g = data[:,:] == 0
# print(len(data[g])/(row*col))
probabilitys =[]
for i in tqdm.tqdm(range(n_classes)):
    probability=0
    for j in range(len(labellist)):
        data = np.load(labellist[j])
        row,col = data.shape
        count = data[:,:] == i
        temp = len(data[count])/(row*col*len(labellist))
        probability += temp
    probabilitys.append(probability)
print(probabilitys)
print(np.sum(np.array(probabilitys)))

name_list = ["其他类别","水田","水浇地","旱耕地","园林","乔木林地","灌木林地","天然草地",
            "人工草地","工业用地","城市住宅","村镇住宅","交通运输","河流","湖泊","坑塘"]
num_list = [1.5,0.6,7.8,6]
plt.bar(range(len(probabilitys)), probabilitys,color='rgb',tick_label=name_list)
plt.show()



