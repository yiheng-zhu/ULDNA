import sys
import os
import evaluation as ev

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

def read_result(result_file):

    f = open(result_file, "r")
    text = f.read()
    f.close()

    return text.splitlines()

def create_result_file(result_dir, label_file, final_result_file):

    name_list = os.listdir(result_dir)
    label_dict = read_label(label_file)

    f = open(final_result_file, "w")
    for name in name_list:
        pro_list = read_result(result_dir + "/" + name)
        label_list = label_dict[name]

        for i in range(len(pro_list)):
            pro = pro_list[i]
            index = str(i+1)
            if(index in label_list):
                label = "1"
            else:
                label = "0"

            f.write(str(pro) + " " + label + "\n")
    f.close()

    roc_file = "roc.txt"

    e = ev.evaluation(final_result_file, roc_file)
    auc = e.process()
    print(auc)

create_result_file(sys.argv[1], sys.argv[2], sys.argv[3])



