import os
import sys

scriptfile = "/data1/zhuyiheng/PDB_library/mmCIF2BioLiP-master/script/cif2pdb"

def create_label(pdb_dir, result_dir, sequence_dir, label_dir, pdb_id):

    pdb_file = pdb_dir + "/" + pdb_id + ".cif.gz"
    if(os.path.exists(pdb_file)):
        os.system("gunzip " + pdb_file)

    pdb_file = pdb_dir + "/" + pdb_id + ".cif"
    result_file = result_dir + "/" + pdb_id



    sequence_file = sequence_dir + "/" + pdb_id + ".fasta"
    label_file = label_dir + "/" + pdb_id

    if(os.path.exists(label_file)):
        return

    cmd = scriptfile + " " + pdb_file + " " + result_file
    os.system(cmd)

    extract_result(result_file + ".txt", pdb_id, sequence_file, label_file)



def extract_result(result_file, pdb_id, sequence_file, label_file):

    if(os.path.exists(result_file)==False or os.path.getsize(result_file)==0):
        return

    f = open(result_file, "r")
    text = f.read()
    f.close()

    index1 = text.find("pmid:")
    index2 = text.find("#recCha")

    text1 = text[index1:index2].strip()
    text2 = text[index2:].strip()

    sequence_dict = dict()

    line_set1 = text1.splitlines()[1:]
    for line in line_set1:
        line = line.strip()
        if(line.startswith(">")):
            values = line.split("\t")
            chain_id = values[0][1:]
            type = values[1].strip()

        else:
            if(type=="protein"):
                name = ">" + pdb_id.upper() + "_" + chain_id
                sequence_dict[name] = line

    label_dict = dict()
    line_set2 = text2.splitlines()[1:]
    for line in line_set2:
        line = line.strip()
        values = line.split("\t")
        chain_id = ">" + pdb_id.upper() + "_" + values[0]
        type = values[1]

        if(type == "dna"):
            label_list = values[5].split()
            for i in range(len(label_list)):

                label_index = int(label_list[i][1:])
                aa = label_list[i][0]
                if(aa!=sequence_dict[chain_id][label_index-1]):
                    print(pdb_id)
                    print(aa)
                    print(sequence_dict[chain_id][label_index-1])
                    print(label_index)
                    print(chain_id)


                label_list[i] = label_index

            if(chain_id not in label_dict):
                label_dict[chain_id] = []
            label_dict[chain_id].extend(label_list)

    f1 = open(sequence_file, "w")
    f2 = open(label_file, "w")

    for chain_id in label_dict:
        label_dict[chain_id] = sorted(list(set(label_dict[chain_id])))

        f1.write(chain_id + "\n" + sequence_dict[chain_id] + "\n")

        line = ""
        for label in label_dict[chain_id]:
            line = line + str(label) + " "
        line = line.strip()

        f2.write(chain_id + "\n" + line + "\n")

    f1.close()
    f2.close()

if __name__ == '__main__':

    pdb_list = os.listdir(sys.argv[1])
    for name in pdb_list:

        create_label(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], name[0:4])




