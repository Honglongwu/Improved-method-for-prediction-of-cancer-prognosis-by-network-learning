# Improved-method-for-prediction-of-cancer-prognosis-by-network-learning


## Welcome to our Git Repository!

## 1.setting

  Python  3.6.3
  
  numpy 1.14.2
  
  sklearn 0.19.1
  
  scipy 1.0.0
  
  tensorflow 1.6.0
  

## 2.  Data
   #### 1> Download URL
: https://www.dropbox.com/sh/tp70gitmmtaft0l/AABLSniRI9lCo1ZqtUGL4ZOqa?dl=0
  
   #### 2>
   ##### (1) mRNA
        :Comma-delimited file of gene expression data 
           Ex)  ,patient1, patient2, patient3
               gene1,-4.556,-1.784,2.295
               gene2,-1.923,1.603,-2.696         
   ##### (2) CNA
         :Comma-delimited file of copy number data 
          Ex)  , patient1, patient2, patient3
              gene1,-0.536,-0.464,8.025
              gene2,7.022,-1.033,-0.636        
   ##### (3) METHYLATION
          :Comma-delimited file of DNA methylation data
           Ex)  , patient1, patient2, patient3
               gene1,7.356,6.404,2.305
               gene2,1.002,3.082,0.006           
   ##### (4) SNP
          :Comma-delimited file of somatic mutation data
           Ex)  , patient 1, patient 2, patient 3
               gene1,0,1,0
               gene2,0,0,4
              
   ##### (5) CLINICAL_FILE
          : Comma-delimited file of Patient's names and osevent
          : Lable 0 = patient who has good prognosis
          : Lable 1 = patient who has bad prognosis 
           Ex) patient1,0
               patient2,1
               patient3,0     
  ##### (6) NETWORK
          : Comma-delimited file of functional interaction network
           EX) GENE,GENE
              gene1, gene2
              gene1, gene3
              gene4, gene5
         
## 3. Run
   ##### python ProposedMethod.py [-t top_n_gene_in_ttest][-i n_experiment][-n n_gene][-d dampingfactor][-l limit_of_experiment] mRNA CNA METHYLATION SNP CLINICAL_FILE NETWORK
  
    
    - top_n_gene_in_ttest : Parameter of step 1. Top N genes show statistical differences between good samples and poor samples in t-test. ( Default: 400 )
    - n_experiment : Parameter of step 2 and step 3 (t in paper). To select a stable and robust feature for random initialization of weights, experiment t times repeatedly in GANs and PageRank step. ( Default : 5 )
    - n_gene : Parameter of step 3 . The number of biomarkers selected for each experiment. ( Default: 250 )
    - dampingfactor : Parameter of step 3. This is damping factor using in PageRank algorithm ( Default: 0.7 )
    - limit_of_experiment : Parameter of step 2,3 (b in paper). when step 2 and step 3 are experimented t times repeatedly, the genes that appear b times in t experiment are selected as biomarkers. The b is the limit of experiment. ( Default: 5 )

###  For Example> 
#### python ProposedMethod.py BRCA_mRNA.txt BRCA_CNA.txt BRCA_methylation.txt BRCA_SNP.txt BRCA_Clinical.txt FIsnetwork.txt

    
    1.preprocessing data...
     loading data...
     divide samples for 10fold validation
     0fold ttest start
     1fold ttest start
     2fold ttest start
     3fold ttest start
     4fold ttest start
     5fold ttest start
     6fold ttest start
     7fold ttest start
     8fold ttest start
     9fold ttest start
    ----------------------------------------------------------------------------------------------------
    2. Step 1 : reconstructing FIs network
    ----------------------------------------------------------------------------------------------------
    3. Step 2,3 : Learning the network and Feature selection using PageRank
     start process  process number :  0     fold number : 0
     start process  process number :  1     fold number : 0
     start process  process number :  2     fold number : 0
     start process  process number :  3     fold number : 0
     start process  process number :  4     fold number : 0
     process 0 converge  Epoch: 0002 n_iter : 0137 D_loss : -0.6936 G_loss : -0.6928
     process 1 converge  Epoch: 0002 n_iter : 0137 D_loss : -0.6937 G_loss : -0.6926
     process 3 converge  Epoch: 0002 n_iter : 0137 D_loss : -0.6934 G_loss : -0.693
     process 2 converge  Epoch: 0002 n_iter : 0137 D_loss : -0.6935 G_loss : -0.6928
     process 4 converge  Epoch: 0002 n_iter : 0137 D_loss : -0.6937 G_loss : -0.6928
     start process  process number :  0     fold number : 1
     start process  process number :  1     fold number : 1
     start process  process number :  2     fold number : 1
     start process  process number :  3     fold number : 1
     start process  process number :  4     fold number : 1
     process 4 converge  Epoch: 0002 n_iter : 0138 D_loss : -0.6934 G_loss : -0.6933
     process 3 converge  Epoch: 0002 n_iter : 0138 D_loss : -0.6933 G_loss : -0.6931
     process 2 converge  Epoch: 0002 n_iter : 0138 D_loss : -0.6932 G_loss : -0.6934
     process 0 converge  Epoch: 0002 n_iter : 0138 D_loss : -0.6935 G_loss : -0.693
     process 1 converge  Epoch: 0002 n_iter : 0138 D_loss : -0.6934 G_loss : -0.6931
     start process  process number :  0     fold number : 2
     start process  process number :  1     fold number : 2
     start process  process number :  2     fold number : 2
     start process  process number :  3     fold number : 2
     start process  process number :  4     fold number : 2
     process 1 converge  Epoch: 0002 n_iter : 0138 D_loss : -0.6932 G_loss : -0.6932
     process 2 converge  Epoch: 0002 n_iter : 0138 D_loss : -0.6934 G_loss : -0.693
     process 4 converge  Epoch: 0002 n_iter : 0138 D_loss : -0.6935 G_loss : -0.693
     process 0 converge  Epoch: 0002 n_iter : 0138 D_loss : -0.693 G_loss : -0.6934
     process 3 converge  Epoch: 0002 n_iter : 0138 D_loss : -0.6934 G_loss : -0.693
     start process  process number :  0     fold number : 3
     start process  process number :  1     fold number : 3
     start process  process number :  2     fold number : 3
     start process  process number :  3     fold number : 3
     start process  process number :  4     fold number : 3
     process 0 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6931 G_loss : -0.6932
     process 1 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6926 G_loss : -0.6937
     process 2 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6926 G_loss : -0.6937
     process 3 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6927 G_loss : -0.6936
     process 4 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6931 G_loss : -0.6932
     start process  process number :  0     fold number : 4
     start process  process number :  1     fold number : 4
     start process  process number :  2     fold number : 4
     start process  process number :  3     fold number : 4
     start process  process number :  4     fold number : 4
     process 3 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6922 G_loss : -0.6941
     process 4 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6928 G_loss : -0.6936
     process 2 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6927 G_loss : -0.6936
     process 0 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6921 G_loss : -0.6942
     process 1 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6925 G_loss : -0.6939
     start process  process number :  0     fold number : 5
     start process  process number :  1     fold number : 5
     start process  process number :  2     fold number : 5
     start process  process number :  3     fold number : 5
     start process  process number :  4     fold number : 5
     process 1 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6937 G_loss : -0.6927
     process 2 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6935 G_loss : -0.6928
     process 0 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6931 G_loss : -0.6932
     process 4 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6932 G_loss : -0.6931
     process 3 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.693 G_loss : -0.6934
     start process  process number :  0     fold number : 6
     start process  process number :  1     fold number : 6
     start process  process number :  2     fold number : 6
     start process  process number :  3     fold number : 6
     start process  process number :  4     fold number : 6
     process 2 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.692 G_loss : -0.6943
     process 4 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6927 G_loss : -0.6936
     process 1 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6923 G_loss : -0.694
     process 0 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6924 G_loss : -0.6939
     process 3 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6923 G_loss : -0.694
     start process  process number :  0     fold number : 7
     start process  process number :  1     fold number : 7
     start process  process number :  2     fold number : 7
     start process  process number :  3     fold number : 7
     start process  process number :  4     fold number : 7
     process 0 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6923 G_loss : -0.6941
     process 3 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6922 G_loss : -0.6942
     process 4 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6929 G_loss : -0.6935
     process 1 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6927 G_loss : -0.6937
     process 2 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6924 G_loss : -0.6939
     start process  process number :  0     fold number : 8
     start process  process number :  1     fold number : 8
     start process  process number :  2     fold number : 8
     start process  process number :  3     fold number : 8
     start process  process number :  4     fold number : 8
     process 3 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6923 G_loss : -0.694
     process 2 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6932 G_loss : -0.6931
     process 4 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6923 G_loss : -0.694
     process 0 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6923 G_loss : -0.694
     process 1 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6928 G_loss : -0.6936
     start process  process number :  0     fold number : 9
     start process  process number :  1     fold number : 9
     start process  process number :  2     fold number : 9
     start process  process number :  3     fold number : 9
     start process  process number :  4     fold number : 9
     process 4 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6934 G_loss : -0.6929
     process 0 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6935 G_loss : -0.6928
     process 1 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6936 G_loss : -0.6928
     process 3 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6931 G_loss : -0.6932
     process 2 converge  Epoch: 0002 n_iter : 0139 D_loss : -0.6935 G_loss : -0.6928
    ----------------------------------------------------------------------------------------------------
    4. Step4 : Prognosis Prediction
     layer= [5]     alpha=  100     10fold AUC=      0.7397354497354498
     layer= [5]     alpha=  150     10fold AUC=      0.7485714285714286
     layer= [5]     alpha=  200     10fold AUC=      0.737989417989418
     layer= [10]    alpha=  100     10fold AUC=      0.7373544973544973
     layer= [10]    alpha=  150     10fold AUC=      0.752962962962963
     layer= [10]    alpha=  200     10fold AUC=      0.7476719576719577

## Output File
  - File name: ProposedMethod_biomarker_per_fold.txt
  - Selected biomarkers of each fold.
  





