import pickle
import sys
import numpy as np
import esm
import torch
import Common_Methods as cm
import os
from script_parameters import depth_threshold

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ''

seed = 123456
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子

# parameters
msa_dir = sys.argv[1]
output_dir = sys.argv[2]
is_random = True

# load MSA transformer
msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer.eval()
msa_batch_converter = msa_alphabet.get_batch_converter()

if torch.cuda.is_available():

    msa_transformer = msa_transformer.cuda()
    print("Transferred model to GPU")

# read dataset

print("The maxnumber of MSA=" + str(depth_threshold))
name_list = os.listdir(msa_dir)


def create_feature_embedding(msa_data):

    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    if torch.cuda.is_available():
        msa_batch_tokens = msa_batch_tokens.to(device="cuda", non_blocking=True)
    with torch.no_grad():
        results = msa_transformer(msa_batch_tokens, repr_layers=[12], return_contacts=True)

    # pred
    feature_embedding = results["representations"][12].to(device="cpu").numpy()

    # save attention map and predicted contact map
    [m, n, p, q] = feature_embedding.shape
    feature_embedding = feature_embedding.reshape(n, p, q)[0][1:, :]

    return feature_embedding


for name in name_list:

    msa_address = msa_dir + "/" + name
    feature_file = output_dir + "/" + name[0: len(name)-4] + ".pt"

    if(os.path.exists(feature_file)):
        continue

    # read msa
    max_depth = len(open(msa_address).readlines()) / 2
    if max_depth < depth_threshold:
        msa_depth = int(max_depth)
    else:
        msa_depth = depth_threshold

    if (is_random):
        msa_data = [cm.read_msa_random(msa_address, msa_depth), ]
    else:
        msa_data = [cm.read_msa(msa_address, msa_depth), ]

    length = len(msa_data[0][0][1])
    cut_off = 1020

    print(name + " " + str(length))

    if(length>1020):

        msa_data1 = []
        msa_data2 = []

        msa_array = msa_data[0]
        for element in msa_array:
            name, sequence = element

            msa_data1.append((name, sequence[0:cut_off]))
            msa_data2.append((name, sequence[cut_off: ]))

        msa_data1 = [msa_data1, ]
        msa_data2 = [msa_data2, ]

        feature_embedding1 = create_feature_embedding(msa_data1)
        feature_embedding2 = create_feature_embedding(msa_data2)

        feature_embedding = np.concatenate((feature_embedding1, feature_embedding2), axis=0)

    else:
        feature_embedding = create_feature_embedding(msa_data)

    print(feature_embedding.shape)

    f = open(os.path.join(feature_file), 'wb')
    pickle.dump(feature_embedding, f)
    f.close()












