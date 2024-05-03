import sys

import numpy as np
import random
import torch
import pickle

tokens = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def aa_code(sequence):

    m = len(sequence)
    n = len(tokens)

    aa_matrix = np.zeros([m, n])
    for i in range(m):
        index = tokens.index(sequence[i])
        aa_matrix[i][index] = 1

    return aa_matrix

def read_name_list(name_file):  # read name list from file

    f = open(name_file, "r")
    text = f.read()
    f.close()
    name_list = np.array(text.splitlines())

    return name_list

def read_label(label_file):  # read label

    f = open(label_file, "r")
    text = f.read()
    f.close()

    label_dict = dict()

    for line in text.splitlines():
        line = line.strip()
        if (line.startswith(">")):
            name = line[1:]
        else:
            label_dict[name] = line.split()

    return label_dict

def read_sequence(sequence_file):

    sequence_dict = dict()

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    for line in text.split():
        line = line.strip()
        if(line.startswith(">")):
            name = line[1:]
        else:
            sequence_dict[name] = line

    return sequence_dict

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

def read_feature_afs(feature_dir, name):

    feature_file_asa = feature_dir + "/asa/" + name + ".asa"
    feature_asa = np.loadtxt(feature_file_asa)

    feature_file_ss = feature_dir + "/ss/" + name + ".ss"
    feature_ss = np.loadtxt(feature_file_ss)

    return np.concatenate((feature_asa, feature_ss), axis=1)


def change(label_index, length):  # change label

    label_array = np.zeros([length, 2])

    for i in range(len(label_array)):
        label_array[i][1] = 1

    for index in label_index:
        label_array[int(index) - 1][0] = 1
        label_array[int(index) - 1][1] = 0

    return label_array


def create_weight(label_index, length):  # change label

    weight_array = np.array([1.0 for i in range(length)])

    weight = (float(length))/float(len(label_index))

    for index in label_index:
        weight_array[int(index) - 1] = weight

    return weight_array

def create_batch(name_list, is_shuffle):  #create batch

    number = len(name_list)
    index = [i for i in range(number)]
    
    if(is_shuffle):
        random.shuffle(index)

    batch_name_list = name_list[index]

    return batch_name_list

def create_data_set(workdir, data_type, is_shuffle):  # create batch name list

    name_list = read_name_list(workdir + "/" + data_type + "_name_list")
    label_dict = read_label(workdir + "/" + data_type + "_label")
    sequence_dict = read_sequence(workdir + "/" + data_type + "_sequence.fasta")

    batch_name_list = create_batch(name_list, is_shuffle)


    return batch_name_list, sequence_dict, label_dict

def read_data_single(feature_dir1, feature_dir2, feature_dir3, feature_dir4, name, sequence_dict, label_dict):  # read feature, label, name

    feature_file1 = feature_dir1 + "/" + name + ".pt"
    feature_file2 = feature_dir2 + "/" + name + ".pt"
    feature_file3 = feature_dir3 + "/" + name + ".pt"

    feature1 = read_feature(feature_file1, 36)
    feature2 = read_feature_protrans(feature_file2)
    feature3 = read_feature_msa(feature_file3)
    feature4 = read_feature_afs(feature_dir4, name)


    label = change(label_dict[name], feature1.shape[0])
    weight = create_weight(label_dict[name], feature1.shape[0])

    return name, feature1, feature2, feature3, feature4, label, weight



if __name__ == '__main__':

    workdir = sys.argv[1]
    data_type = "train"
    is_shuffle = True
    feature_dir1 = "/data1/zhuyiheng/DNA/features/"
    feature_dir2 = "/data1/zhuyiheng/DNA/features_protrans/"
    feature_dir3 = "/data1/zhuyiheng/DNA/msa_features_256/"
    feature_dir4 = "/data1/zhuyiheng/DNA/dssp_feature/"

    batch_name_list, sequence_dict, label_dict = create_data_set(workdir, data_type, is_shuffle)
    for name in batch_name_list:

        train_name, feature1, feature2, feature3, feature4, label, weight = read_data_single(feature_dir1, feature_dir2, feature_dir3, feature_dir4, name, sequence_dict, label_dict)
        print(name)
        print(feature1.shape)
        print(feature2.shape)
        print(feature3.shape)
        print(feature4.shape)

        print()

        break




