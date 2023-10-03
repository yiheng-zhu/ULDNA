# ULDNA
ULDNA is a protein-DNA binding site predictor through integrating protein language models with LSTM-attention network.

1. Install Softwares   
(a) Install ESM2 transformer and ESM-MSA transformer, see details in https://github.com/facebookresearch/esm.   
(b) Install ProtTrans transformer, see details in https://github.com/agemagician/ProtTrans.   
(c) Install HHblits software, see details in https://github.com/soedinglab/hh-suite/wiki#hh-suite-databases.  
(d) Download Uniclust30 database, see details in https://gwdu111.gwdg.de/~compbiol/uniclust/2023_02/.

2. Training Model   
(a) Extract feature embeddings from ESM2 transformer  
python Extract_FE_ESM2.py ESM_Model_Name Sequence_File Feature_Embedding_Dir --repr_layers layer_number --include per_tok
e.g., python Extract_FE_ESM2.py esm2_t36_3B_UR50D ./sequence.fasta ./esm_feature2/ --repr_layers 36 --include per_tok
    
(b) Extract feature embeddings from ProtTrans transformer
python ./training/Extract_FE_ProtTrans.py sequence_file feature_embedding_dir
 

