import random
import numpy as np
import tqdm
import os
import argparse
import tifffile as tiff

def get_labels():
    """Load the mapping that associates classes with label colors

    Returns:
        np.ndarray with dimensions (13, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],  # 0，其他类别
            [0, 200, 0],  # 1，水田
            [150, 250, 0],  # 2，水浇地
            [150, 200, 150],  # 3，旱耕地
            [200, 0, 200],  # 4，园林
            [150, 0, 250],  # 5，乔木林地
            [150, 150, 250],  # 6，灌木林地
            [250, 200, 0],  # 7，天然草地
            [200, 200, 0],  # 8，人工草地
            [200, 0, 0],  # 9，工业用地
            [250, 0, 150],  # 10，城市住宅
            [200, 150, 150],  # 11，村镇住宅
            [250, 150, 150],  # 12，交通运输
            [0, 0, 200],  # 13，河流
            [0, 150, 200],  # 14，湖泊
            [0, 200, 250],  # 15，坑塘
        ]
    )


    # ["其他类别","水田","水浇地","旱耕地","园林","乔木林地","灌木林地","天然草地",
    #  "人工草地","工业用地","城市住宅","村镇住宅","交通运输","河流","湖泊","坑塘"]


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes

    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.

    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(np.uint8)
    return label_mask


def decode_segmap(label_mask, n_classes):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)


def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def get_rand_patch(img, mask, sz=160):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz and img.shape[0:2] == mask.shape[0:2]
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]

    # Apply some random transformations
    random_transformation = np.random.randint(1,9)
    if random_transformation == 1:  # reverse first dimension
        patch_img = patch_img[::-1,:,:]
        patch_mask = patch_mask[::-1,:,:]
    elif random_transformation == 2:    # reverse second dimension
        patch_img = patch_img[:,::-1,:]
        patch_mask = patch_mask[:,::-1,:]
    elif random_transformation == 3:    # transpose(interchange) first and second dimensions
        patch_img = patch_img.transpose([1,0,2])
        patch_mask = patch_mask.transpose([1,0,2])
    elif random_transformation == 4:
        patch_img = np.rot90(patch_img, 1)
        patch_mask = np.rot90(patch_mask, 1)
    elif random_transformation == 5:
        patch_img = np.rot90(patch_img, 2)
        patch_mask = np.rot90(patch_mask, 2)
    elif random_transformation == 6:
        patch_img = np.rot90(patch_img, 3)
        patch_mask = np.rot90(patch_mask, 3)
    elif random_transformation == 7:
        patch_img = add_noise(patch_img)
    else:
        pass

    return patch_img, patch_mask


def get_patches(x_dict, y_dict, aug_class, saveDir, n_classes=16, n_patches=1000, sz=256):
    img_path = os.path.join(saveDir,"train")
    mask_path = os.path.join(saveDir,"train_label")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    for i in tqdm.tqdm(range(len(aug_class))):
        total_patches = 0
        while total_patches < n_patches:
            img_id = random.sample(x_dict.keys(), 1)[0]
            img = x_dict[img_id]
            mask = y_dict[img_id]
            img_patch, mask_patch = get_rand_patch(img, mask, sz)
            encode_mask = encode_segmap(mask_patch)
            class_list = np.unique(encode_mask[50:sz-50,50:sz-50])
            if aug_class[i] in class_list:
                imgName = os.path.join(img_path,f"{aug_class[i]}_{total_patches}.npy")
                maskName = os.path.join(mask_path,f"{aug_class[i]}_{total_patches}.npy")
                mask_patch = decode_segmap(encode_mask,n_classes=n_classes)    ################  set  class's number
                np.save(imgName,img_patch)
                np.save(maskName,mask_patch)
                total_patches += 1
                print('Generated {} patches'.format(total_patches))
            else:
                pass



trainIds = [str(i).zfill(2) for i in range(1, 9)]  # all availiable ids: from "01" to "24"


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--n_patches", type=int, default=1600,help='the number of picture in every class ')
    parse.add_argument("--patches_size", type=int, default=256,help='the size of images')
    parse.add_argument("--n_classes", type=int, default=16)
    parse.add_argument("--saveDir", type=str, default="./data_random_augmentation")
    args = parse.parse_args()

    aug_class = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading images')
    for img_id in trainIds:
        img_m = np.array(tiff.imread('./train/{}.tif'.format(img_id)))
        mask = np.load('./train_label_npy/{}.npy'.format(img_id))
        X_DICT_TRAIN[img_id] = img_m
        Y_DICT_TRAIN[img_id] = mask
        print(img_id + ' read')
    print('Images were read')
    get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, aug_class, args.saveDir, args.n_classes, args.n_patches, args.patches_size)
