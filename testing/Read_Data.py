import numpy as np
import torch
import pickle

from script_parameters import sub_feature_path1, sub_feature_path2, sub_feature_path3, test_file, esm2_layer, feature_size4, label_size

def read_feature(feature_file, index):   # read features from transformer

    feature = torch.load(feature_file)
    feature = feature["representations"][index].numpy()
    feature = feature.astype(np.float64)

    return feature

def read_feature_msa(feature_file):


    feature = pickle.load(open(feature_file, 'rb'))
    feature = feature.astype(np.float64)

    return feature



def read_feature_protrans(feature_file):

    feature = torch.load(feature_file)
    feature = feature.astype(np.float64)

    return feature

def get_name_list(name_file):

    f = open(name_file, "rU")
    text = f.read()
    f.close()

    name_list = []

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):
            name_list.append(line[1:])

    return name_list


def read_data_single(feature_dir1, feature_dir2, feature_dir3, name):  # read feature, label, name

    feature_file1 = feature_dir1 + "/" + name + ".pt"
    feature_file2 = feature_dir2 + "/" + name + ".pt"
    feature_file3 = feature_dir3 + "/" + name + ".pt"

    feature1 = read_feature(feature_file1, esm2_layer)
    feature2 = read_feature_protrans(feature_file2)
    feature3 = read_feature_msa(feature_file3)
    feature4 = np.zeros([feature1.shape[0], feature_size4])
    label = np.zeros([feature1.shape[0], label_size])
    weight = np.array([1 for i in range(feature1.shape[0])])

    return feature1, feature2, feature3, feature4, label, weight

if __name__ == '__main__':

    name_list = get_name_list(test_file)
    for name in name_list:

        feature1, feature2, feature3, feature4, label, weight = read_data_single(sub_feature_path1, sub_feature_path2, sub_feature_path3)

        print(feature1.shape)
        print(feature2.shape)
        print(feature3.shape)
        print(feature4.shape)
        print(label.shape)
        print(weight.shape)

        print()






