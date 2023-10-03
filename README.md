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
    e.g., python ./training/Extract_FE_ESM2.py esm2_t36_3B_UR50D ./sequence.fasta ./esm2_feature/ --repr_layers 36 --include per_tok
                   
    (b) Extract feature embeddings from ProtTrans transformer   
    python ./training/Extract_FE_ProtTrans.py sequence_file feature_embedding_dir   
   e.g., python ./training/Extract_FE_ProtTrans.py ./sequence.fasta ./prottrans_feature/  

    (c) Create MSA for query sequences
    python ./training/Create_MSA.py sequence_file msa_dir  
    e.g., python ./training/Create_MSA.py ./sequence.fasta ./msa_workspace/

    (d) Extract feature embeddings from EMS-MSA transformer  
    python ./training/Extract_FE_MSA.py msa_dir feature_embedding_dir  
    e.g., python ./training/Extract_FE_MSA.py ./msa_workspace/msa/ ./msa_feature/

    (e) Training models  
    Python ./training/training_model.py workdir round_time, GPU_id, GPU_ratio, max_iteration, is_used_ESM2, is_used_ProtTrans, is_used_EMS_MSA  
    e.g., Python ./training/training_model.py ./PDNA-543/Independent/ 1 1 1.0 10 1 1 1  
    * workdir should contain train_sequence.fasta, train_name_list, train_label, test_sequence.fasta, test_name_list, and test_label.    
    * we implement each model in ten round, where round_time means the current round.
    * is_used_ESM2 = 1 means that we use the feature embeddings of ESM2 transformer to training models.
    * is_used_ProTrans = 1 means that we use the feature embeddings of ESM2 transformer to training models.
    * is_used_ESM-MSA = 1 means that we use the feature embeddings of ESM2 transformer to training models.

3. testing model
    python ./testing/test_model.py test_file model_type threshold result_dir  
    python ./testing/test_model.py ./test.fasta PDNA-543 0.5 ./test_result/
    * we provide two models which are trained on PDNA-543 and PDNA-335 datasets, respectively, see details in ./model/     
    
 

