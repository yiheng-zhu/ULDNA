
home_path = "/data1/webserver/webserver/zhuyiheng/ULDNA/"

code_path = home_path + "/code/"

feature_path = home_path + "/features/"
sub_feature_path1 = feature_path + "/esm2/"
sub_feature_path2 = feature_path + "/protrans/"
sub_feature_path3 = feature_path + "/ems_msa/"

test_file = home_path + "/test.fasta"

esm2_file = code_path + "/extract.py"
esm2_model = "esm2_t36_3B_UR50D"
esm2_layer = 36

msa_path = home_path + "/MSA/"
hhblits_database = home_path + "/UniRef30_2022_02/UniRef30_2022_02"
hhblits_bindir = "/usr/opt/hhsuite/bin/"

depth_threshold = 256

feature_size1 = 2560
feature_size2 = 1024
feature_size3 = 768
feature_size4 = 18
label_size = 2

model_dir = home_path + "/Model/"
result_dir = home_path + "/Result/"
final_result_file = home_path + "/results.csv"
average_result_dir = result_dir + "/final/"

model_number = 10
PDNA_543_CT = [0.265, 0.240, 0.075]
PDNA_335_CT = [0.295, 0.315, 0.105]

model_type_file = home_path + "/model_type"


