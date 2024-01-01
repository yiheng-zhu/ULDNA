import math
import os
import sys
from Bio.PDB import *

code_standard = \
    {
    'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
    'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
    'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
    'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G',
    #'MSE':'M',
    'DA':'A', 'DC':'C', 'DG':'G', 'DT':'T',
    'A':'A', 'C':'C', 'G':'G', 'T':'T', "U": "U"
    }

radim = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "P": 1.8, "S": 1.8, "F": 1.47, "I": 1.98}

cut_off = 3.5
max_length = 1000
min_length = 30


def judge_protein(sequence):

    s_set = set(sequence)
    dna_set = {"A", "T", "G", "C"}
    if (len(s_set - dna_set) == 0):
        return False

    return True

def ed_distance(v1, v2):

    return math.sqrt((v1[0]-v2[0]) * (v1[0]-v2[0]) + (v1[1]-v2[1]) * (v1[1]-v2[1]) + (v1[2]-v2[2]) * (v1[2]-v2[2]))

def judge_binding(residue_atom_name, base_atom_name, v1, v2):

    if(ed_distance(v1, v2) <=cut_off):
        return True
    else:
        return False

def label_site_one(resiude_atom_list, dna_atom_list):

    count = 0

    for residue_atom in resiude_atom_list:

        residue_atom_name = residue_atom.get_name()
        v1 = residue_atom.get_vector()

        for base_atom_list in dna_atom_list:
            for base_atom in base_atom_list:
                base_atom_name = base_atom.get_name()
                v2 = base_atom.get_vector()

                if(judge_binding(residue_atom_name, base_atom_name, v1, v2)==True):
                    count = count + 1

                if(count>=1):
                    return True

    return False


def label_site(protein_atom_list, dna_atom_list):

    label_list = []

    for i in range(len(protein_atom_list)):
        residue_atom_list = protein_atom_list[i]

        if(label_site_one(residue_atom_list, dna_atom_list)==True):
            label_list.append(i+1)

    return label_list





def create_single_label(pdb_file, pdb_id):

    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)

    model = structure.get_models()
    models = list(model)
    print("模型的数量：", end="")
    print(len(models))

    chain_list = list(models[0].get_chains())
    print("链的数量：", end="")
    print(len(chain_list))

    protein_dict = dict()
    protein_dict["seq"] = dict()
    protein_dict["atom"] = dict()
    protein_dict["label"] = dict()

    dna_dict = dict()
    dna_dict["seq"] = dict()
    dna_dict["atom"] = dict()

    for chain in chain_list:

        chain_id = chain.get_id()

        sequence = ""
        atoms_list = []
        for residue in chain:
            if (residue.get_resname() == "HOH" or residue.id[0]!=" "):
                continue
            else:
                res_name = residue.get_resname()
                if(res_name not in code_standard):
                    sequence = sequence + "X"
                else:
                    sequence = sequence + code_standard[res_name]

                atoms_list.append(list(residue.get_atoms()))


        if (judge_protein(sequence)):
            protein_dict["seq"][chain_id] = sequence
            protein_dict["atom"][chain_id] = atoms_list

        else:
            dna_dict["seq"][chain_id] = sequence
            dna_dict["atom"][chain_id] = atoms_list


    if (len(protein_dict) > 0 and len(dna_dict) > 0):

        protein_chain_list = protein_dict["atom"].keys()
        dna_chain_list = dna_dict["atom"].keys()


        for protein_chain_id in protein_chain_list:



            protein_atom_list = protein_dict["atom"][protein_chain_id]
            label_list = []

            if (len(protein_dict["seq"][protein_chain_id]) > max_length or len(protein_dict["seq"][protein_chain_id]) < min_length):
                protein_dict["label"][protein_chain_id] = label_list
                continue

            for dna_chain_id in dna_chain_list:
                dna_atom_list = dna_dict["atom"][dna_chain_id]
                label_list.extend(label_site(protein_atom_list, dna_atom_list))

            label_list = sorted(list(set(label_list)))

            protein_dict["label"][protein_chain_id] = label_list

            print(protein_chain_id)
            print(label_list)


    return protein_dict


def create_labels(pdb_dir, pdb_name_list, sequence_file, label_file):

    f1 = open(sequence_file, "w")
    f2 = open(label_file, "w")

    f = open(pdb_name_list, "r")
    text = f.read()
    f.close()

    name_list = text.splitlines()

    for name in name_list:

        pdb_file = pdb_dir + "/pdb" + name + ".ent"
        print(pdb_file)
        if(os.path.exists(pdb_file)==False):
            continue

        pdb_id = name
        protein_dict = create_single_label(pdb_file, pdb_id)

        chain_id_list = protein_dict["seq"].keys()
        for chain_id in chain_id_list:
            if(len(protein_dict["label"][chain_id])>0):
                f1.write(">" + pdb_id.upper() + "_" + chain_id + "\n" + protein_dict["seq"][chain_id] + "\n")

                label_line = ""
                for label in protein_dict["label"][chain_id]:
                    label_line = label_line + str(label) + " "
                label_line = label_line.strip()
                f2.write(">" + pdb_id.upper() + "_" + chain_id + "\n" + label_line + "\n")

                f1.flush()
                f2.flush()

    f1.close()
    f2.close()

if __name__ == '__main__':

    create_labels(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
