import os.path
import random
import sys
import numpy as np

def read_sequence(sequence_file):

    sequence_dict = dict()

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):
            name = line[1:]
        else:
            sequence_dict[name] = line

    return sequence_dict

def read_label(label_file):

    label_dict = dict()

    f = open(label_file, "r")
    text = f.read()
    f.close()

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):
            name = line[1:]
        else:
            label_dict[name] = line.split()

    return label_dict


def select_dict(name_list, origin_dict):

    deal_dict = dict()
    for name in name_list:
        deal_dict[name] = origin_dict[name]

    return deal_dict

def write_name(name_list, name_file):

    f = open(name_file, "w")
    for name in name_list:
        f.write(name + "\n")
    f.close()

def write_sequence(sequence_dict, sequence_file):

    f = open(sequence_file, "w")
    for name in sequence_dict:
        f.write(">" + name + "\n")
        f.write(sequence_dict[name] + "\n")
    f.close()

def write_label(label_dict, label_file):

    f = open(label_file, "w")
    for name in label_dict:
        line = ""
        for label in label_dict[name]:
            line = line + label + " "
        f.write(">" + name + "\n" + line + "\n")
    f.close()

def split(sequence_file, label_file, outputdir, cross_number):

    sequence_dict = read_sequence(sequence_file)
    label_dict = read_label(label_file)

    name_list = np.array(sorted(sequence_dict.keys()))

    number = len(name_list)
    index = [i for i in range(number)]

    random.shuffle(index)

    sub_number = int(number/cross_number)
    batch_name_list = []

    for i in range(cross_number-1):
        batch_name_list.append(name_list[index[int(i*sub_number):int((i+1)*sub_number)]])

    batch_name_list.append(name_list[index[int((i+1) * sub_number):]])

    for i in range(cross_number):

        test_name_list = batch_name_list[i]
        test_sequence_dict = select_dict(test_name_list, sequence_dict)
        test_label_dict = select_dict(test_name_list, label_dict)

        train_name_list = list(set(name_list)-set(test_name_list))
        train_sequence_dict = select_dict(train_name_list, sequence_dict)
        train_label_dict = select_dict(train_name_list, label_dict)

        current_dir = outputdir + "/cross" + str(i+1) + "/"
        if(os.path.exists(current_dir)==False):
            os.makedirs(current_dir)

        train_name_file = current_dir + "/train_name_list"
        test_name_file = current_dir + "/test_name_list"
        write_name(train_name_list, train_name_file)
        write_name(test_name_list, test_name_file)

        train_sequence_file = current_dir + "/train_sequence.fasta"
        test_sequence_file = current_dir + "/test_sequence.fasta"
        write_sequence(train_sequence_dict, train_sequence_file)
        write_sequence(test_sequence_dict, test_sequence_file)

        train_label_file = current_dir + "/train_label"
        test_label_file = current_dir + "/test_label"
        write_label(train_label_dict, train_label_file)
        write_label(test_label_dict, test_label_file)


split(sys.argv[1], sys.argv[2], sys.argv[3], 10)





