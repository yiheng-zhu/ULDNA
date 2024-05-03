import os
import random

import evaluation as ev
import Get_Meatures_From_T as gt
import numpy as np

from scipy.spatial.distance import cdist

def read_name_list(name_file):

    f = open(name_file, "r")
    text = f.read()
    f.close()

    return text.splitlines()


def EuclideanDistances(A, B):

    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)

    return np.array(ED)

def save_results(workdir, method, data_type, round_times, index, name, score, label):

    resultdir = workdir + "/" + method + "_result/" + data_type + "/round" + str(round_times) + "/" + str(index) + "/"

    if(os.path.exists(resultdir) == False):
        os.makedirs(resultdir)

    result_file = resultdir + "/" + name

    f = open(result_file, "w")
    for i in range(len(score)):
        f.write(str(score[i][0]) + " " + str(label[i][0]) + "\n")
    f.close()

def combine_result(workdir, method, data_type, round_times, index):

    name_list = read_name_list(workdir + "/" + data_type + "_name_list")

    resultdir = workdir + "/" + method + "_result/" + data_type + "/round" + str(round_times) + "/" + str(index) + "/"

    combine_file = resultdir + "/result"
    f1 = open(combine_file, "w")

    for name in name_list:
        f2 = open(resultdir + "/" + name, "r")
        text = f2.read()
        f2.close()

        f1.write(text)
    f1.close()

    rocdir = workdir + "/" + method + "_roc/" + data_type + "/round" + str(round_times) + "/"
    if (os.path.exists(rocdir) == False):
        os.makedirs(rocdir)
    roc_file = rocdir + "/roc" + str(index)
    e = ev.evaluation(combine_file, roc_file)
    auc = e.process()

    return roc_file, auc

def write_measure(record_file, index, measure_list, data_type, is_start, is_end, auc, postfix):  # write measure

    f = open(record_file, "a")
    if(is_start):
        line = "The " + str(index) + "-th iteration: " + postfix + "\n"
        f.write(line)
    line = data_type + ": "
    for measure in measure_list:
        line = line + measure + " "
    line = line.strip() + " AUC=" + str(auc) + "\n"
    if(is_end):
        line = line + "\n"

    f.write(line + "\n")
    f.flush()
    f.close()

def create_cross_entropy_result(workdir, method, data_type, round_times, index, postfix):

    roc_file1, auc1 = combine_result(workdir, method, data_type, round_times, index)
    opt_t1 = gt.find_opt_fmax(roc_file1)
    measure_list1 = gt.get_fmax_by_T(roc_file1, opt_t1)

    record_file = workdir + "/record" + str(round_times)

    write_measure(record_file, index, measure_list1, "test", True, True, auc1, postfix)


def create_sample_index(train_label):

    all_number = train_label.shape[0]
    ratio = 4


    label_dim1 = np.array(train_label[:, 0])
    neg_index = []
    for i in range(len(label_dim1)):
        if(label_dim1[i] == 0):
            neg_index.append(i)

    neg_index = np.array(neg_index)
    neg_number = len(neg_index)
    pos_number = all_number - neg_number
    all_neg_index = [i for i in range(neg_number)]
    random.shuffle(all_neg_index)

    select_number = pos_number * ratio
    delete_neg_index = neg_index[all_neg_index[select_number:]]

    bias = [0 for i in range(all_number)]
    for i in range(len(bias)):
        if(i in delete_neg_index):
            bias[i] = -10

    return bias






def calculate_triplet_result(train_output, test_output, train_label, test_label, select_number, result_file, distance_file):

    #distance = np.matmul(test_output, np.transpose(train_output))
    distance =  EuclideanDistances(test_output, train_output)
    #print(distance)
    #distance = distance + create_sample_index(train_label)
    distance_index = np.argsort(distance, axis = 1)

    test_residue_number = test_label.shape[0]

    f = open(result_file, "w")

    for i in range(test_residue_number):

        real_label = test_label[i][0]

        sum1 = 0.0
        sum2 = 0.0

        for j in range(select_number):
            weight = (select_number - j + 0.0)/select_number
            sum1 = sum1 + weight
            train_index = distance_index[i][j]
            if(int(train_label[train_index][0])==1):
                sum2 = sum2 + weight

        score = sum2/sum1
        f.write(str(score) + " " + str(real_label) + "\n")

    f.close()

    sub_distance_index = distance_index[:, 0:200]
    np.savetxt(distance_file, sub_distance_index, "%d")
    np.savetxt(distance_file + "_train_label", train_label, "%d")

def save_triplet_result(workdir, method, data_type, round_times, index, train_output, test_output, test_name, train_label, test_label, select_number):

    resultdir = workdir + "/" + method + "_result/" + data_type + "/round" + str(round_times) + "/" + str(index) + "/"
    if(os.path.exists(resultdir)==False):
        os.makedirs(resultdir)

    result_file = resultdir + "/" + test_name

    distance_dir = workdir + "/" + method + "_distance/" + data_type + "/round" + str(round_times) + "/" + str(index) + "/"
    if (os.path.exists(distance_dir) == False):
        os.makedirs(distance_dir)
    distance_file = distance_dir + "/" + test_name
    calculate_triplet_result(train_output, test_output, train_label, test_label, select_number, result_file, distance_file)


def create_triplet_result(workdir, method, data_type, round_times, index, postfix):

    roc_file1, auc1 = combine_result(workdir, method, data_type, round_times, index)
    opt_t1 = gt.find_opt_fmax(roc_file1)
    measure_list1 = gt.get_fmax_by_T(roc_file1, opt_t1)

    record_file = workdir + "/triplet_record" + str(round_times)
    write_measure(record_file, index, measure_list1, "test", True, True, auc1, postfix)















