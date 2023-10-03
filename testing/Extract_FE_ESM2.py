import os
import sys
from script_parameters import esm2_file, esm2_model, esm2_layer

sequence_file = sys.argv[1]
output_dir = sys.argv[2]

cmd = "python " + esm2_file + " " + esm2_model + " " + \
      sequence_file + " " + output_dir + " --repr_layers " + str(esm2_layer) + " --include per_tok"

os.system(cmd)