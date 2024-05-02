# ULDNA
ULDNA is a protein-DNA binding site predictor through integrating protein language models with LSTM-attention network.  
The webserver of ULDNA is freely available at https://csbioinformatics.njust.edu.cn/uldna/.  
Note:  
(1) The web server only accepts proteins with a length less than 1000. If the length of the query protein sequence is larger than 1000, please split it into multiple protein sequences.  
(2) Due to limited computational resources, the number of submitted proteins  cannot exceed 10. Do not submit the next task while the previous prediction task is still in progress.  
(3) If you have any questions regarding tasks, please send an email to yihzhu@njau.edu.cn (Ph.D. Yi-Heng Zhu). 

Reference.
Yi-Heng Zhu, Zi Liu, Zhiwei Ji*, Dong-Jun Yu*. ULDNA: Integrating Unsupervised Multi-Source Language Models with LSTM-Attention Network for High-Accuracy Protein-DNA Binding Site Prediction. Briefings in Bioinformatics. 2024, 25(2):bbae040. 


 

1. Install softwares   
    (a) Install ESM2 transformer and ESM-MSA transformer, see details in https://github.com/facebookresearch/esm.  
    Note: You should modify the source codes of modules.py and msa_transformer.py for ESM-MSA, which are located in xxx/anaconda3/lib/pythonxxx/site-packages/esm/.  In modules.py, you should modify the codes from line 367 to line 386 (class ContactPredictionHead(nn.Module)). In msa_transformer.py, you should modify the codes from line 206 to line 241 (def forward()). The modified modules.py and msa_transformer.py can be found in ./ESM-MSA_Modification/.
   
    (b) Install ProtTrans transformer, see details in https://github.com/agemagician/ProtTrans.  
    Note: If you use the new version of ProtTrans updated after 2023/07/14, the “Extract_FE_ProtTrans.py” may have errors. You should use the "prott5_embedder.py (downloaded from https://github.com/agemagician/ProtTrans)" to extract feature embeddings for proteins sequences.
   
    (c) Install HHblits software, see details in https://github.com/soedinglab/hh-suite/wiki#hh-suite-databases.  
    (d) Download Uniclust30 database, see details in https://gwdu111.gwdg.de/~compbiol/uniclust/2023_02/.

2. Prediction    
   We use the downloaded model to predict DNA bindings sites from protein sequences.  
   (a) Download models from https://csbioinformatics.njust.edu.cn/uldna/model.zip.  
   (b) Python ./testing/test_model.py test_file model_type threshold  
       e.g., python test_model.py test.fasta PDNA-543 0.5  
   * we provide two models which are trained on PDNA-543 and PDNA-335 datasets, respectively, see details in ./model/   
   * The ULDNA Model only accepts proteins with a length less than 1000. If the length of the query protein sequence is larger than 1000, please split it into multiple protein sequences.
    
3. Training models (Optional)  
    (a) Extract feature embeddings from ESM2 transformer  
    python Extract_FE_ESM2.py sequence_File Feature_embedding_Dir   
    e.g., python ./training/Extract_FE_ESM2.py ./sequence.fasta ./esm2_feature/ 
                   
    (b) Extract feature embeddings from ProtTrans transformer   
    python ./training/Extract_FE_ProtTrans.py sequence_file feature_embedding_dir   
    e.g., python ./training/Extract_FE_ProtTrans.py ./sequence.fasta ./prottrans_feature/  

    (c) Create MSA for query sequences
    python ./training/Create_Single_HHM.py sequence_file msa_dir  
    e.g., python ./training/Create_Single_HHM.py ./sequence.fasta ./msa_workspace/

    (d) Extract feature embeddings from EMS-MSA transformer  
    python ./training/Extract_FE_MSA.py msa_dir feature_embedding_dir  
    e.g., python ./training/Extract_FE_MSA.py ./msa_workspace/msa/ ./msa_feature/

    (e) Training models  
    Python ./training/training_model.py workdir round_time, GPU_id, GPU_ratio, max_iteration, is_used_ESM2, is_used_ProtTrans, is_used_EMS_MSA  
    e.g., Python ./training/training_model.py ./example/ 1 1 1.0 10 1 1 1  
    * workdir should contain train_sequence.fasta, train_name_list, train_label, test_sequence.fasta, test_name_list, and test_label.    
    * We implement each model in ten round, where round_time means the current round.
    * is_used_ESM2 = 1 means that we use the feature embeddings of ESM2 transformer to training models.
    * is_used_ProTrans = 1 means that we use the feature embeddings of ProtTrans transformer to training models.
    * is_used_ESM-MSA = 1 means that we use the feature embeddings of ESM-MSA transformer to training models.


      
    
4. Cross-Validation (Optional)   
    python Split_Cross.py sequence_file label_file output_dir, cross_number  
    e.g., python Split_Cross.py ./sequence.fasta ./sequence_label ./PDNA-543/Cross_Validation/ 10

5. Evaluation (Optional) 
    python evalution.py result_file, roc_file  
    e.g., python evalution.py ./result.txt ./roc.txt  

