import numpy as np
import random
import os

def read_msa_random(msa_file, max_depth):

    f = open(msa_file, "r")
    text = f.read()
    f.close()

    sequence = ""
    name = ""
    name_list = []
    sequence_dict = dict()

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):

            if(len(sequence)>0):

                name_list.append(name)
                sequence_dict[name] = remove_gap_sequence(sequence)

            name = line
            sequence = ""

        else:
            sequence = sequence + line

    name_list.append(name)
    sequence_dict[name] = remove_gap_sequence(sequence)

    final_name_list = []
    sub_name_list = name_list[1:]
    random.shuffle(sub_name_list)
    sub_name_list = sub_name_list[:max_depth-1]

    final_name_list.append(name_list[0])
    final_name_list.extend(sub_name_list)

    final_sequence_list = []
    final_sequence_list.append(sequence_dict[name_list[0]])
    for name in sub_name_list:
        final_sequence_list.append(sequence_dict[name])

    return [(final_name_list[i], final_sequence_list[i]) for i in range(len(final_name_list))]

def read_length(length_file):

    f = open(length_file, "r")
    text = f.read()
    f.close()

    length_dict = dict()
    for line in text.splitlines():
        values = line.strip().split()
        length_dict[values[0]] = values[1:]

    return length_dict

def create_dir(workdir):
    if (os.path.exists(workdir) == False):
        os.makedirs(workdir)


def max_matrix(a, b):

    m, n =  a.shape
    c = np.array(np.zeros([m,n]))
    for i in range(m):
        for j in range(n):
            c[i, j] = max(a[i,j], b[i,j])

    return c

def is_lower(s):

    ap_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

    return s not in ap_list

def do_filter(sequence):

    sequence = "".join(list(filter(is_lower, sequence)))
    return sequence


def remove_gap_sequence(sequence):

    sequence = do_filter(sequence)

    return sequence