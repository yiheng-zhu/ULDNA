import os
import sys

from script_parameters import code_path, feature_path, msa_path, \
    sub_feature_path1, sub_feature_path2, sub_feature_path3

os.system("rm -rf " + feature_path)
os.system("rm -rf " + msa_path)

os.makedirs(sub_feature_path1)
os.makedirs(sub_feature_path2)
os.makedirs(sub_feature_path3)
os.makedirs(msa_path)

test_file = sys.argv[1]
model_type = sys.argv[2]
threshold = float(sys.argv[3])

os.system("rm -rf " + feature_path)
script_file1 = code_path + "/Extract_FE_ESM2.py"
cmd1 = "python " + script_file1 + " " + test_file + " " + sub_feature_path1
os.system(cmd1)

script_file2 = code_path + "/Extract_FE_ProtTrans.py"
cmd2 = "python " + script_file2 + " " + test_file + " " + sub_feature_path2
os.system(cmd2)

script_file3 = code_path + "/Create_Single_HHM.py"
cmd3 = "python " + script_file3 + " " + test_file + " " + msa_path
os.system(cmd3)

script_file4 = code_path + "/Extract_FE_MSA.py"
cmd4 = "/data/zhuyiheng/anaconda3/envs/ULDNA/bin/python " + script_file4 + " " + msa_path + "/msa/" + " " + sub_feature_path3
os.system(cmd4)

script_file5 = code_path + "/Load_Model.py"
cmd5 = "python " + script_file5 + " " + test_file + " " + model_type
os.system(cmd5)

script_file6 = code_path + "/Create_Result.py"
cmd6 = "python " + script_file6 + " " + model_type + " " + str(threshold)
os.system(cmd6)


