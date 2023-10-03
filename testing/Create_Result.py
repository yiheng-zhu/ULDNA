import os

import numpy as np
import sys
from decimal import Decimal
from script_parameters import final_result_file, model_number, PDNA_543_CT, PDNA_335_CT, average_result_dir
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

def get_name_list(sequence_file):

    f = open(sequence_file, "rU")
    text = f.read()
    f.close()

    name_list = []

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):
            name_list.append(line[1:])

    return name_list

def read_result(result_file):

    f = open(result_file, "r")
    text = f.read()
    f.close()

    pro_list = []
    for line in text.splitlines():
        pro_list.append(float(line.strip()))

    return np.array(pro_list)

def create_combine_file(sequence_file, result_dir, combine_result_file, times, cut_off):


    sequence_dict = read_sequence(sequence_file)
    name_list = get_name_list(sequence_file)

    os.system("rm -rf " + average_result_dir)
    os.makedirs(average_result_dir)

    f = open(combine_result_file, "w")

    for name in name_list:

        line = "Prot.ID,NO,Residue,DNA-Binding Probability,DNA-Binding Result,\n"
        f.write(line)

        result_file = result_dir + "/1/" + name
        pro_list = read_result(result_file)

        for i in range(2, times+1):
            pro_list = pro_list + read_result(result_dir + "/" + str(i) + "/" + name)
        pro_list = pro_list/times

        for i in range(len(pro_list)):
            id = str(i+1).zfill(4)
            res = sequence_dict[name][i]
            pro = str(Decimal(pro_list[i]).quantize(Decimal("0.000000")))

            if(pro_list[i]>=cut_off):
                label = "B"
            else:
                label = "N"

            f.write(name + "," + id + "," + res + "," + pro + "," + label + "\n")

        f1 = open(average_result_dir + "/" + name, "w")

        for pro in pro_list:
            f1.write(str(Decimal(pro).quantize(Decimal("0.000000"))) + "\n")
        f1.close()

    f.close()

if __name__ == '__main__':

    test_file = sys.argv[1]
    model_type = sys.argv[2]
    threshold = sys.argv[3]
    result_dir = sys.argv[4]


    if(model_type=="PDNA-543"):
        if (threshold == "t1"):
            cut_off = PDNA_543_CT[0]
        if (threshold == "t2"):
            cut_off = PDNA_543_CT[1]
        if (threshold == "t3"):
            cut_off = PDNA_543_CT[2]
    else:
        if (threshold == "t1"):
            cut_off = PDNA_335_CT[0]
        if (threshold == "t2"):
            cut_off = PDNA_335_CT[1]
        if (threshold == "t3"):
            cut_off = PDNA_335_CT[2]

    create_combine_file(test_file, result_dir, final_result_file, model_number, cut_off)



