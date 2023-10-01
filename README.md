# ULDNA
ULDNA is a protein-DNA binding site predictor through integrating protein language models with LSTM-attention network.

1. Install Softwares   
(a) Install ESM2 transformer and ESM-MSA transformer, see details in https://github.com/facebookresearch/esm.   
(b) Install ProtTrans transformer, see details in https://github.com/agemagician/ProtTrans.   
(c) Install HHblits software, see details in https://github.com/soedinglab/hh-suite/wiki#hh-suite-databases.  
(d) Download Uniclust30 database, see details in https://gwdu111.gwdg.de/~compbiol/uniclust/2023_02/.

2. Feature Embedding Extraction  
(a) Extract feature embeddings for ESM2 transformer  
python extract.py -ESM_Model_Name -Sequence_File -Output_Dir --repr_layers layer_number --include per_tok    
(b)  

