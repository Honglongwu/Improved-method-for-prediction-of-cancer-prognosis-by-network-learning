import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import math
import random
from sklearn.metrics import roc_curve, auc 
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from math import sqrt, ceil
from operator import itemgetter
from copy import deepcopy
from sklearn.model_selection import KFold
from scipy import stats
from multiprocessing import Process, Queue
import sys
import argparse
random.seed(0)
np.random.seed(0)



def main():
  
	#preprocossing.
	pm=preprocessing() 
	print("----------------------------------------------------------------------------------------------------")
	print("2. Step 1 : reconstructing FIs network")
	reconstructed_FIs_perfold,gene_in_reconstructed_FIs_perfold=pm.reconstruct_FIs_network()
	print("----------------------------------------------------------------------------------------------------")
	print("3. Step 2,3 : Learning the network and Feature selection using PageRank")
	biomarker_perfold=[]
	for foldnum in range(10):
		data_for_GANs=pm.mk_data_for_GANs(gene_in_reconstructed_FIs_perfold[foldnum],foldnum)
		
		score=np.zeros((len(pm.mRNA.index)))
		# multiprocessing.
		output1 = Queue(); output2 = Queue(); output3 = Queue();output4 = Queue();output5 = Queue();
		output6 = Queue(); output7 = Queue(); output8 = Queue();output9 = Queue();output10 = Queue();
		output11 = Queue(); output12 = Queue(); output13 = Queue();output14 = Queue();output15 = Queue();
		output16 = Queue(); output17 = Queue(); output18 = Queue();output19 = Queue();output20 = Queue();
		process_list = []
		Output = [output1, output2, output3,output4,output5,output6, output7, output8,output9,output10,output11, output12, output13,output14,output15,output16, output17, output18,output19,output20]
		
		#To select a stable and robust feature for random initialization of weights, repeatedly experiment with the reconstructed network learning-phase using GANs and the PageRank process (t times). 
		#t is n_experiment.
		for process_number in range(pm.n_experiment) :
			process_list.append(Process(target=pm.Learning_FIsnetwork_GANs, args=(process_number, reconstructed_FIs_perfold[foldnum],data_for_GANs,foldnum, Output[process_number])))

		for n,p in enumerate(process_list) :
			p.start()	 

		
		
		result_GANs=[]
		for i in range(pm.n_experiment):
			result_GANs.append(Output[i].get());
		
		for process in process_list :
			process.join()
		#select the genes that appeared more than b times in t experiments as biomarkers.
		#t is n_experiment.
		#b is limit.
		for i in range(pm.n_experiment):
			pagerank_genes=pm.pagerank(result_GANs[i])
			for k in pagerank_genes:
				score[pm.gene2num[k]]=score[pm.gene2num[k]]+1
		biomarker=[]
		for i,j in zip(score,pm.mRNA.index):
			if i >=pm.limit:
				biomarker.append(j)
		biomarker_perfold.append(biomarker)
		
	#save biomarker
	f = open("ProposedMethod_biomarker_per_fold.txt", 'w')
	for foldnum in range(10):
		f.write("\nFold Number(10 fold validation) : %d\n" % foldnum)
		for gene in biomarker_perfold[foldnum]:
			f.write("%s\t" % gene)
	f.close()
	print("----------------------------------------------------------------------------------------------------")
	print("4. Step4 : Prognosis Prediction")
	pm.auc(reconstructed_FIs_perfold,biomarker_perfold)


"""
	From here,Functions for preprocessing
	this step includes loading data,intersectioning data,z-scoring for each sample and t-test for each fold

"""

#read a comma-delimited text file.
def read_file(file):
	with open(file, 'r') as fop :
		data= []
		for line in fop :
			data.append(line.strip().split(','))
	data=np.array(data)
	data=DataFrame(data[1:,1:],columns=data[0,1:],index=data[1:,0])
	return data

#download mRNA,CNA,methylation,SNP,clinical file(lable),FIs network and parameters.
def loading_data():		
	#download mRNA,CNA,methylation and SNP.
	parser=argparse.ArgumentParser(description="Improved method for prediction of cancer prognosis using network and multi-omics data")					
	parser.add_argument('mRNA', type=str, help="Gene expression data")
	parser.add_argument('CNA', type=str, help="Copy number data")
	parser.add_argument('METHYLATION', type=str, help="Methylation data")
	parser.add_argument('SNP', type=str, help="Somatic mutation data")
	parser.add_argument('CLINICAL_FILE', type=str, help="If the patient's label is 0, the patient has a good prognosis.And if the patient's label is 1, the patient has a bad prognosis.")
	parser.add_argument('NETWORK', type=str, help="Functional interaction network.")
	
	parser.add_argument('-t','--top_n_gene_in_ttest',type=int, default=400, help="Parameter of step 1. top N genes with a large difference between good and bad patients in the t-test.(Default is 400)")
	parser.add_argument('-i','--n_experiment',type=int, default=5, help="Parameter of step 2 and step 3 (t in paper). To select a stable and robust feature for weight random initialization, the number of times that appling the pagerank step and learning reconstructed FIs network using GANs step experiment repeatedly.(Default is 5)")
	parser.add_argument('-n','--n_gene',type=int, default=250, help="Parameter of step 3. the number of biomarkers selected for each experiment (Default is 250)")
	parser.add_argument('-d','--dampingfactor',type=float, default=0.7, help="Parameter of step 3. This is damping factor using in pagerank algorithm (Default is 0.7)") 
	parser.add_argument('-l','--limit_of_experiment',type=int, default=5, help="Parameter of step 2,3 (b in paper). when step 2 and step 3 are repeated t times, the genes that appeared b times in t times is selected as biomarkers. The b is the limit of experiment.(Default is 5)") 
	
	mRNA=read_file(parser.parse_args().mRNA)
	CNA=read_file(parser.parse_args().CNA)
	met=read_file(parser.parse_args().METHYLATION)
	snp=read_file(parser.parse_args().SNP)
	
	#download FIS network.
	with open(parser.parse_args().NETWORK, 'r') as fop :
		edges = []
		for line in fop :
			edges.append(line.strip().split(','))
			
	#download lable (lable 0: patient who has good prognosis,lable 1: patient who has bad prognosis).
	with open(parser.parse_args().CLINICAL_FILE, 'r') as fop :
		cli = []
		for line in fop :
			cli.append(line.strip().split(','))
	cli=np.array(cli)
	lable=Series(cli[:,1],index=cli[:,0])
	
	#download parameters
	n_gene_in_ttest=parser.parse_args().top_n_gene_in_ttest
	n_biomarker=parser.parse_args().n_gene
	damping_factor=parser.parse_args().dampingfactor
	n_experiment=parser.parse_args().n_experiment
	n_limit=parser.parse_args().limit_of_experiment
	return mRNA,CNA,met,snp,edges,lable,n_gene_in_ttest,n_biomarker,damping_factor, n_experiment,n_limit

#find the intersection of genes in mRNA, CNA, methylation,SNP data and FIs network and the intersection of samples in mRNA, CNA, methylation, and SNP data.
def intersetion_data(raw_mRNA,raw_CNA,raw_met,raw_snp,raw_edges,raw_clinical_file):
	#find the intersection of genes in mRNA, CNA, methylation, and SNP data.
	co_gene=[x for x in raw_mRNA.index if x in raw_CNA.index]
	co_gene=[x for x in raw_met.index if x in co_gene]
	co_gene=[x for x in raw_snp.index if x in co_gene]

	#find the intersection between the genes from the previous step and the genes in the FIs network.
	edge_list = []
	ppi_genes = set()
	for edge in raw_edges:
		gene1, gene2 = edge[0], edge[1]
		condition = ((gene1 in co_gene) and (gene2 in co_gene))
		if condition :
			edge_list.append([gene1, gene2])
			ppi_genes.add(gene1)
			ppi_genes.add(gene2)
	ppi_genes = list(ppi_genes)

	#find the intersection of samples in mRNA, CNA, methylation, and SNP data.
	co_sample=[x for x in raw_mRNA.columns if x in raw_CNA.columns]
	co_sample=[x for x in raw_met.columns if x in co_sample]
	co_sample=[x for x in raw_snp.columns if x in co_sample]

	#modify raw mRNA, raw CNA, raw methylation, raw SNP, and raw lable data with the intersection of the genes and the intersection of the samples.
	mRNA=raw_mRNA.loc[ppi_genes,co_sample]
	CNA=raw_CNA.loc[ppi_genes,co_sample]
	met=raw_met.loc[ppi_genes,co_sample]
	snp=raw_snp.loc[ppi_genes,co_sample]
	lable=raw_clinical_file.loc[co_sample]
	
	return mRNA,CNA,snp,met,edge_list,lable	

#normalizing data for each sample by z-scoring.
def zscore(data) :
	len_row_gene = data.shape[0]
	len_column_sample = data.shape[1]
	zscored_data = np.zeros((len_row_gene, len_column_sample))
	for column_sample in range(len_column_sample):
		mu = data[:,column_sample].mean()
		sigma = data[:,column_sample].std()
		if mu!=0 and sigma!=0:
			for row_gene in range(len_row_gene):
				x = data[row_gene][column_sample]
				zscored_data[row_gene][column_sample] = (x - mu)/sigma
		else:
			print('Warning!z-scoring!')

	return zscored_data
	
#seperate patients who have bad prognosis and patients who have good prognosis.	
def seperate_good_bad_patients(sampleList,lable):
	good_samples=[]
	bad_samples=[]
	for i,j in zip(sampleList,lable[sampleList]):
		if j=='0':
			good_samples.append(i)
		elif j=='1':
			bad_samples.append(i)
		else:
			#exception Handling
			print('#########################################################################################################################')
			print('									error!lable can be only 0 or 1')
			print('									You have to stop this process!')
			print('#########################################################################################################################')
	return good_samples,bad_samples



#perfom t-test comparing patients who have poor prognosis and patients who have good prognosis for each dataset.
#num2gene is Series in which data is a gene and index is a gene number.
#good_sam is the list of samples which have good prognosis and bad_sam is the list of samples which have bad prognosis. 
def t_test(mRNA,CNA,met,snp,num2gene,good_sam,bad_sam):
		
		#exception Handling
		if len(good_sam)==0:
			print('#########################################################################################################################')
			print('									Error!there is no good prognostic patient')
			print('									You have to stop this process!')
			print('#########################################################################################################################')
		if len(bad_sam)==0:
			print('#########################################################################################################################')
			print('									error!there is no bad prognostic patient')
			print('										You have to stop this process!')
			print('#########################################################################################################################')
			
			
		n_genes=len(mRNA.index)
		genes=mRNA.index
		
		#perform a t-test for each gene in mRNA data conparing a poor prognostic patient group and a good prognostic patient group. 
		t_scores = np.zeros(n_genes, dtype=np.float32)	
		for i in range(n_genes):
			poor_data = mRNA.loc[num2gene[i], bad_sam].values.astype(np.float64)
			good_data = mRNA.loc[num2gene[i], good_sam].values.astype(np.float64)
			t_statistic = abs(stats.ttest_ind(poor_data,good_data)[0])
			if np.isnan(t_statistic) :
				t_statistic = 0
				t_scores[i] = t_statistic
			else :
				t_scores[i] = t_statistic
		
		#t_scores is DataFrame in which data is a list of the t-test statistics in mRNA data and index is a list of genes. 
		t_scores=DataFrame(t_scores,index=genes)
		
		#perform a t-test for each gene in CNA data conparing a poor prognosis patient group and a good prognosis patient group.
		t_scores2 = np.zeros(n_genes, dtype=np.float32)	
		for i in range(n_genes):
			poor_data = CNA.loc[num2gene[i], bad_sam].values.astype(np.float64)
			good_data = CNA.loc[num2gene[i], good_sam].values.astype(np.float64)
			t_statistic = abs(stats.ttest_ind(poor_data,good_data)[0])
			if np.isnan(t_statistic) :
				t_statistic = 0
				t_scores2[i] = t_statistic
			else :
				t_scores2[i] = t_statistic
		#t_scores2 is DataFrame in which data is a list of the t-test statistics in CNA data and index is a list of genes.  
		t_scores2=DataFrame(t_scores2,index=genes)
		
		#Perform a t-test for each gene in methylation data conparing a poor prognosis patient group and a good prognosis patient group.
		t_scores3= np.zeros(n_genes, dtype=np.float32)	
		for i in range(n_genes):
			poor_data = met.loc[num2gene[i], bad_sam].values.astype(np.float64)
			good_data = met.loc[num2gene[i], good_sam].values.astype(np.float64)
			t_statistic = abs(stats.ttest_ind(poor_data,good_data)[0])
			if np.isnan(t_statistic) :
				t_statistic = 0
				t_scores3[i] = t_statistic
			else :
				t_scores3[i] = t_statistic
		#t_scores3 is DataFrame in which data is a list of the t-test statistics in methylation data and index is a list of genes. 
		t_scores3=DataFrame(t_scores3,index=genes)
		
		#Perform a t-test for each gene in SNP data conparing a poor prognosis patient group and a good prognosis patient group.
		t_scores4= np.zeros(n_genes, dtype=np.float32)	
		for i in range(n_genes):
			poor_data = snp.loc[num2gene[i], bad_sam].values.astype(np.float64)
			good_data = snp.loc[num2gene[i], good_sam].values.astype(np.float64)
			t_statistic = abs(stats.ttest_ind(poor_data,good_data)[0])
			if np.isnan(t_statistic) :
				t_statistic = 0
				t_scores4[i] = t_statistic
			else :
				t_scores4[i] = t_statistic
		#t_scores4 is DataFrame in which data is a list of the t-test statistics in SNP data and index is a list of genes.
		t_scores4=DataFrame(t_scores4,index=genes)
		
		return t_scores,t_scores2,t_scores3,t_scores4

#perform preprocessing.		
def preprocessing():
	print('1.preprocessing data...')
	#download mRNA,CNA,methylation,SNP,lable,FIs network data and parameters.
	print(' loading data...')
	raw_mRNA,raw_CNA,raw_met,raw_snp,raw_edges,raw_lable,n_gene_in_ttest,n_biomarker,damping_factor, n_experiment,n_limit=loading_data()
	
	
	
	#find the intersection of genes in mRNA, CNA, methylation,SNP data and FIs network and the intersection of samples from mRNA, CNA, methylation, and SNP data.
	#then modify raw mRNA, raw CNA, raw methylation, raw SNP, and raw lable data with the intersection of the genes and the intersection of the samples.
	
	raw_mRNA2,raw_CNA2,raw_met2,raw_snp2,edge_list,lable=intersetion_data(raw_mRNA,raw_CNA,raw_met,raw_snp,raw_edges,raw_lable)
	
	#normalizing data for each sample by z-scoring in mRNA, CNA, methylation and SNP data respectly.
	mvalues=zscore(raw_mRNA2.values.astype('float64'))
	CNAvalues=zscore(raw_CNA2.values.astype('float64'))
	metvalues=zscore(raw_met2.values.astype('float64'))
	snpvalues=zscore(raw_snp2.values.astype('float64'))
	mRNA = DataFrame(mvalues, index=raw_mRNA2.index, columns=raw_mRNA2.columns)
	CNA = DataFrame(CNAvalues, index=raw_CNA2.index, columns=raw_CNA2.columns)
	met = DataFrame(metvalues, index=raw_met2.index, columns=raw_met2.columns)
	snp = DataFrame(snpvalues, index=raw_snp2.index, columns=raw_snp2.columns)
	
	#gene2num and num2gene are for mapping between genes and numbers.
	gene2num = {}
	num2gene = {}
	for i, gene in enumerate(mRNA.index):
		gene2num[gene] = i
		num2gene[i] = gene
		
	#divide samples for 10fold validation 
	print(' divide samples for 10fold validation ')
	good_sam,bad_sam=seperate_good_bad_patients(mRNA.columns,lable)
	good_sam=np.array(good_sam)
	bad_sam=np.array(bad_sam)
	kf = KFold(n_splits=10, random_state=None, shuffle=False)

	good_train_samples=[]
	bad_train_samples=[]
	test_samples=[]
	for good_index, bad_index in zip(kf.split(good_sam),kf.split(bad_sam)):
		good_train, good_test = good_sam[good_index[0]], good_sam[good_index[1]]
		bad_train, bad_test = bad_sam[bad_index[0]], bad_sam[bad_index[1]]
		good_train_samples.append(good_train)
		bad_train_samples.append(bad_train)
		test_tmp=np.hstack((good_test,bad_test))
		test_samples.append(test_tmp)

	
	#perform a t-test for each fold.
	mRNA_ttest=[]
	CNA_ttest=[]
	met_ttest=[]
	snp_ttest=[]
	for foldnum in range(10):
		print(' '+str(foldnum)+'fold ttest start')
		goodsam=good_train_samples[foldnum]
		badsam=bad_train_samples[foldnum]
		mRNA_ttmp,CNA_ttmp,met_ttmp,snp_ttmp=t_test(mRNA,CNA,met,snp,num2gene,goodsam,badsam)
		mRNA_ttest.append(mRNA_ttmp)
		CNA_ttest.append(CNA_ttmp)
		met_ttest.append(met_ttmp)
		snp_ttest.append(snp_ttmp)

	
	#make instance of class PM.
	Pm=PM(n_gene_in_ttest,n_biomarker,damping_factor, n_experiment,n_limit,mRNA,CNA,met,snp,lable,edge_list,good_train_samples,bad_train_samples,test_samples,gene2num,num2gene,mRNA_ttest,CNA_ttest,met_ttest,snp_ttest)
	
	return Pm

"""
	From here,Functions for Step 1,2,3 and 4 in paper.
	step1 is recostructing FIs network.
	step2 is learning the network.
	step3 is Feature selection using PageRank.
	step4 is prognosis predicition.
"""	

#PM is the class which has multi-omics Data, parameters, series to map between genes and gene numbers, samples for 10 fold validation , t-statistics for each fold and functions for Step 1, 2, 3, and 4.
class PM:

	#initialize variables.
	"""
		self.n_gene_in_ttest is a parameter which is the number of genes which have high absolute values of t-statistics to be used in step 1.
		self.n_experiment is parameter which is the number of times to repeat step2 and step3.
		self.n_biomarker is parameter which is the number of genes which are selected as biomarkers.
		self.limit is parameter of step 2,3. when step2 and step3 are repeated t times(t=n_experiment), the genes that appeared b times in t times is selected as biomarkers. The b is the limit.
		self.damping_factor is damping factor using in pagerank algorithm.
		self.mRNA is mRNA data.
		self.CNA is CNA data.
		self.met is methylation data.
		self.snp is SNP data.
		self.lable is lable of samples (clinical data).
		self.edge_list is edges in FIs network.
		self.good_train_samples is a list containing lists of good prognostic training samples in each fold for 10 fold validation.
		self.bad_train_samples is a list containing lists of bad prognostic trainging samples in each fold for 10 fold validation.
		self.test_samples a list containing lists of test samples in each fold for 10 fold validation.
		self.gene2num is series for mapping from genes to gene numbers.
		self.num2gene is series for mapping from gene numbers to genes.
		self.mRNA_ttest is a list containing DataFrames that are the results of t-test in mRNA data per fold at the preprocessing stage.
		self.CNA_ttest is a list containing DataFrames that are the results of t-test in CNA data per fold at the preprocessing stage.
		self.met_ttest is a list containing DataFrames that are the results of t-test in methylation data per fold at the preprocessing stage.
		self.snp_ttest is a list containing DataFrames that are the results of t-test in SNP data per fold at the preprocessing stage.
	"""
	def __init__(self,n_gene_in_ttest,n_biomarker,damping_factor, n_experiment,n_limit,mRNA,CNA,met,snp,lable,edge_list,good_train_samples,bad_train_samples,test_samples,gene2num,num2gene,mRNA_ttest,CNA_ttest,met_ttest,snp_ttest):
		self.n_gene_in_ttest=n_gene_in_ttest
		self.n_experiment=n_experiment
		self.n_biomarker=n_biomarker
		self.limit=n_limit
		self.damping_factor=damping_factor
		self.mRNA = mRNA
		self.CNA=CNA
		self.met=met
		self.snp=snp
		self.lable=lable
		self.edge_list=edge_list
		self.good_train_samples=good_train_samples
		self.bad_train_samples=bad_train_samples
		self.test_samples=test_samples
		self.gene2num=gene2num
		self.num2gene=num2gene
		self.mRNA_ttest=mRNA_ttest
		self.CNA_ttest=CNA_ttest
		self.met_ttest=met_ttest
		self.snp_ttest=snp_ttest
	
	#step 1. reconstruct FIs network.
	def reconstruct_FIs_network(self):
		#reconstructed_network_10fold is list containing lists of the edges in reconstructed network per fold.
		reconstructed_network_10fold=[]
		
		#gene_in_reconstructed_network_10fold is list containing sets of the genes in reconstructed network per fold.
		gene_in_reconstructed_network_10fold=[]
		
		#reconstruct network per fold.
		for foldnum in range(10):
			
			#reconstructed_network is a list of the edges in reconstructed network in the fold (foldnum :fold number for 10fold validation).
			reconstructed_network=[]
			
			#gene_in_reconstructed_network is a set of the genes in reconstructed network in the fold (foldnum :fold number for 10fold validation).
			gene_in_reconstructed_network=set()
		
			#sort genes by t-statistics.			
			mRNA_t_sort=self.mRNA_ttest[foldnum].sort_values(by=0,ascending=False)
			CNA_t_sort=self.CNA_ttest[foldnum].sort_values(by=0,ascending=False)
			met_t_sort=self.met_ttest[foldnum].sort_values(by=0,ascending=False)
			snp_t_sort=self.snp_ttest[foldnum].sort_values(by=0,ascending=False)
			
			#selected_by_ttest is a set of the top N genes which have high absolute values of t-statistics in mRNA, CNA, methylation, or SNP data.
			selected_by_ttest=set()
			selected_by_ttest.update(mRNA_t_sort.index[:self.n_gene_in_ttest])
			selected_by_ttest.update(CNA_t_sort.index[:self.n_gene_in_ttest])
			selected_by_ttest.update(met_t_sort.index[:self.n_gene_in_ttest])
			selected_by_ttest.update(snp_t_sort.index[:self.n_gene_in_ttest])
			
			#construct a network. include all edges involving at least one of the genes belonging to selected_by_ttest.
			for edge in self.edge_list:
				if edge[0] in selected_by_ttest or edge[1] in selected_by_ttest:
					reconstructed_network.append(edge)
					gene_in_reconstructed_network.update(edge)
			reconstructed_network_10fold.append(reconstructed_network)
			gene_in_reconstructed_network_10fold.append(gene_in_reconstructed_network)
		return reconstructed_network_10fold,gene_in_reconstructed_network_10fold
		
	#step 2-1. make data for GANs. 
	#foldnum is fold number.
	#network is the gene in reconstructed network in the fold.
	def mk_data_for_GANs(self,networkgene,foldnum):
		
		#merge traing samples. 
		trainsample=np.hstack((self.good_train_samples[foldnum],self.bad_train_samples[foldnum]))
		random.seed(0) 
		
		#to suffle between train samples have good prognosis and train sample have bad prognosis.
		random.shuffle(trainsample)
		
		#result_tmp is list containing the data for GANs.
		result_tmp=[]
		
		#for each gene in reconstructed network, select the dataset with the largest absolute value of t-test statistic.
		for j in networkgene:
			genevec=[self.mRNA_ttest[foldnum].loc[j,0],self.CNA_ttest[foldnum].loc[j,0],self.met_ttest[foldnum].loc[j,0],self.snp_ttest[foldnum].loc[j,0]]
			num=np.argmax(genevec)
			if num==0:
				result_tmp.append(self.mRNA.loc[j,trainsample].values.astype('float64'))
			elif num==1:
				result_tmp.append(self.CNA.loc[j,trainsample].values.astype('float64'))
			elif num==2:
				result_tmp.append(self.met.loc[j,trainsample].values.astype('float64'))
			elif num==3:
				result_tmp.append(self.snp.loc[j,trainsample].values.astype('float64'))
		return result_tmp
	
	#step 2-2. learn reconstructed FIs network using GANs.
	#process_number is process number in multiprocessing. 
	#edge_list is the edges of FIs network in the fold.
	#data_for_GANs is the data we made in step 2-1.
	#foldnum is the fold number.
	def Learning_FIsnetwork_GANs(self,process_number,edge_list,data_for_GANs,foldnum,output):
		
		#creat an adjacency matrix from the reconstructed FIs network.
		def make_adjacencyMatrix_for_GANs(n_genes,edge_list):
			matrix = np.zeros([n_genes,n_genes], dtype=np.float32)
			for edge in edge_list:
				x = gene2num_forGANs[edge[0]]
				y = gene2num_forGANs[edge[1]]
				matrix[x][y] = matrix[y][x] = 1.
			return matrix
		
		#prepare for GANs.
		def prepare(adjacency_matrix,n_input,n_hidden,n_noise,stddev):
			reconstucted_network_adjacency_matrix = tf.constant(adjacency_matrix)
			
			#input.
			X = tf.placeholder(tf.float32, [None, n_input])
			
			#noise for generator.
			Z = tf.placeholder(tf.float32, [None, n_noise])
			#G_W is for generator.
			G_W = tf.Variable(tf.random_normal([n_noise, n_genes], stddev=0.01))
			
			#D_W1 is for discriminator.
			D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
			
			#D_W2 is for discriminator.
			D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
			
			return reconstucted_network_adjacency_matrix,X,Z,G_W,D_W1,D_W2
		
		#generator of GANs.
		def generator(G_W,reconstucted_network_adjacency_matrix,noise_z):
			output = tf.nn.relu(tf.matmul(noise_z, reconstucted_network_adjacency_matrix*(G_W*tf.transpose(G_W))))	   
			return output
			
		#discriminator of GANs.
		def discriminator(inputs,D_W1,D_W2):
			hidden = tf.nn.relu(tf.matmul(inputs, D_W1))
			output = tf.nn.sigmoid(tf.matmul(hidden, D_W2))
			return output
			
		#make random variables for generator.
		def get_noise(batch_size, n_noise):
			return np.random.normal(size=(batch_size, n_noise))
			
		print(' start process	process number : ',process_number,'	fold number :',foldnum)
		
		#get a set of genes from reconstructed FIs network.
		total_gene=[]
		for i in edge_list:
			total_gene.append(i[0])
			total_gene.append(i[1])
		total_gene=set(total_gene)
		
		#make series to map between genes and genes number only for GANs.
		gene2num_forGANs = {}
		num2gene_forGANs = {}
		for i, gene in enumerate(total_gene):
			gene2num_forGANs[gene] = i
			num2gene_forGANs[i] = gene	
		
		#n_genes is the length of genes from the reconstructed FIs network.
		n_genes = len(total_gene)
		
		data_for_GANs=np.array(data_for_GANs)
		data_for_GANs = data_for_GANs.T
		
		#creat an adjacency matrix from the reconstructed FIs network.
		adjacency_matrix = make_adjacencyMatrix_for_GANs(n_genes,edge_list)
		
		#set the parameters.		
		tf.set_random_seed(process_number)
		batch_size = 1
		learning_rate = 0.0002

		
		#reconstucted_network_adjacency_matrix is an adjacency matrix of the reconstructed FIs network.
		reconstucted_network_adjacency_matrix,X,Z,G_W,D_W1,D_W2=prepare(adjacency_matrix,n_genes,256,n_genes,0.01)

		G = generator(G_W,reconstucted_network_adjacency_matrix,Z)

		D_gene = discriminator(G,D_W1,D_W2)

		D_real = discriminator(X,D_W1,D_W2)
		
		#loss function.
		loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))

		loss_G = tf.reduce_mean(tf.log(D_gene))

		D_var_list = [D_W1, D_W2]
		G_var_list = [G_W]

		#define optimizer.
		train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
		train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)
		

		n_iter = data_for_GANs.shape[0]
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		loss_val_D, loss_val_G = 0, 0

		
		#perform GANs.
		for epoch in range(2):
			loss_val_D_list = []
			loss_val_G_list = []
			for i in range(n_iter):
				batch_xs = data_for_GANs[i].reshape(1,-1)
				noise = get_noise(1, n_genes)
	  
				_, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
				_, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})
				loss_val_D_list.append(loss_val_D)
				loss_val_G_list.append(loss_val_G)	
			
					
		print(' process '+str(process_number)+' converge ','Epoch:', '%04d' % (epoch+1),'n_iter :', '%04d' % n_iter,'D_loss : {:.4}'.format(np.mean(loss_val_D_list)),'G_loss : {:.4}'.format(np.mean(loss_val_G_list)))
				
		
		network = sess.run(reconstucted_network_adjacency_matrix*(G_W*tf.transpose(G_W)))
		
		#rearrange the result of GANs.
		result = []
		for n in range(0, len(network)):
			for m in range(n+1, len(network[n])):
				val = network[n][m]
				if val != 0 :
					result.append([num2gene_forGANs[n], num2gene_forGANs[m], val])
	  
		output.put(result)
	
	#function for pagerank.
	#weight is obtained from GANs.
	def pagerank(self,weight):
	
		

		damping_factor=self.damping_factor
		threshold=0.005
		
		#make weighted graph with GANs weight.
		def mk_graph_with_weight(n_genes,weight):
			matrix = np.zeros([n_genes,n_genes], dtype=np.float32)
			count=0
			for edge in weight:
				x = self.gene2num[edge[0]]
				y = self.gene2num[edge[1]]
				matrix[x][y] =abs(float(edge[2]))
				matrix[y][x] =abs(float(edge[2]))
			result = np.zeros(matrix.shape, dtype=np.float32)
			n_genes = matrix.shape[0]
			for i in range(n_genes):
				z =matrix[:,i].sum()
				if z > 0.:
					result[:,i] = matrix[:,i] / z
			return result


		n_genes=len(self.mRNA.index)
		result_mat = mk_graph_with_weight(n_genes,weight)

		#step 2. pageRank.
		def perform_pagerank(matrix, d, threshold):
			n_genes = matrix.shape[0]
			score = np.ones(n_genes) / n_genes
			tmp = np.ones(n_genes) / n_genes

			for i in range(100):
				score = (1.-d)*(np.ones(n_genes) / n_genes) + d*(matrix.dot(score))
				deviation = np.abs(score-tmp).max()
				if deviation < threshold:
					break
				else:
					tmp = deepcopy(score)
			return score
		#perform pagerank.
		score = perform_pagerank(result_mat, damping_factor, threshold)
		
		#sort gene by pagerank score.
		result=DataFrame(score)
		result_gene=result.sort_values(by=0,ascending=False).index
		
		#select biomarkers using pagerank score.
		real_result=[]
		for x in result_gene:  
			real_result.append(self.num2gene[x])
			if len(real_result)==self.n_biomarker:
				break
		return real_result
	
	
	
	#create an adjacency matrix form edge_list.
	def make_adjacencyMatrix(self,edge_list):
		n_genes=len(self.mRNA.index)
		matrix = np.zeros([n_genes,n_genes], dtype=np.float32)
		count=0
		for edge in edge_list:
			x = self.gene2num[edge[0]]
			y = self.gene2num[edge[1]]
			matrix[x][y] =1
			matrix[y][x] =1
		return matrix
	
	#measure the area under the curve(AUC).
	def auc(self,reconstructed_FIs,biomarkerperfold) :
		
		#names is the list containing the classifiers names.
		names = [n for n in range(6)]
		
		#classi is the series in which data are lists containing the AUCs for each fold in each classifier and indexs are the names of the classifiers.
		classi={}
		for i in names:
			classi[i]=list()
		
		for foldnum in range(10):
			#sample is a numpy array of the training samples in the fold (foldnum is fold number for 10 fold validation).
			sample=np.hstack((self.good_train_samples[foldnum],self.bad_train_samples[foldnum]))
			
			#test_sample is a list of test samples in the fold (foldnum is fold number for 10 fold validation).
			test_sample=self.test_samples[foldnum]
			random.shuffle(sample)
			
						
			#edge_list is the list of the edges in reconstructed FIs network of the fold (foldnum is fold number for 10 fold validation).
			edge_list=reconstructed_FIs[foldnum]
			edge_list=np.array(edge_list)



			#from here, make data for AUC.
			
			
			selected_by_ttest=set()

			#sort genes by t-statistics
			mRNA_ttest=self.mRNA_ttest[foldnum].sort_values(by=0,ascending=False)
			CNA_ttest=self.CNA_ttest[foldnum].sort_values(by=0,ascending=False)
			met_ttest=self.met_ttest[foldnum].sort_values(by=0,ascending=False)
			snp_ttest=self.snp_ttest[foldnum].sort_values(by=0,ascending=False)
			
			#mRNA_t is the list of genes which have high absolute values of t-statistics in mRNA data and were used to reconstruct FIs network in step 1.
			mRNA_t=mRNA_ttest.index[:self.n_gene_in_ttest]
			#CNA_t is the list of genes which have high absolute values of t-statistics in CNA data and were used to reconstruct FIs network in step 1.
			CNA_t=CNA_ttest.index[:self.n_gene_in_ttest]
			#met_t is the list of genes which have high absolute values of t-statistics in methylation data and were used to reconstruct FIs network in step 1.
			met_t=met_ttest.index[:self.n_gene_in_ttest]
			#snp_t is the list of genes which have high absolute values of t-statistics in SNP data and were used to reconstruct FIs network in step 1.
			snp_t=snp_ttest.index[:self.n_gene_in_ttest]
			
			#make union of mRNA_t,CNA_t,met_t and snp_t.
			selected_by_ttest.update(mRNA_t)
			selected_by_ttest.update(CNA_t)
			selected_by_ttest.update(met_t)
			selected_by_ttest.update(snp_t)
			
			
			#data_tmp is a list containing a data for training.
			data_tmp=[]
			#test_tmp is a list containg a data for test.
			test_tmp=[]
			
			#create adjacency_matrix from reconstructed FIs network in the fold.
			flag=self.make_adjacencyMatrix(edge_list)
			
			#biomarker is the list of genes which were selected in step 2 and 3 for the fold (foldnum is fold number).
			biomarker=biomarkerperfold[foldnum]

			#decide the dataset per gene.
			for bi in biomarker:
				#if the gene belongs to mRNA_t,CNA_t,met_t or snp_t, select the dataset of mRNA, CNA, methylation or SNP respectly for the gene.
				if bi in selected_by_ttest:
					if bi in mRNA_t:
						data_tmp.append(self.mRNA.ix[bi,sample].values.astype('float64'))
						test_tmp.append(self.mRNA.ix[bi,test_sample].values.astype('float64'))
					if bi in CNA_t:
						data_tmp.append(self.CNA.ix[bi,sample].values.astype('float64'))
						test_tmp.append(self.CNA.ix[bi,test_sample].values.astype('float64'))
					if bi in met_t:
						data_tmp.append(self.met.ix[bi,sample].values.astype('float64'))
						test_tmp.append(self.met.ix[bi,test_sample].values.astype('float64'))
					if bi in snp_t:
						data_tmp.append(self.snp.ix[bi,sample].values.astype('float64'))
						test_tmp.append(self.snp.ix[bi,test_sample].values.astype('float64'))
				else:
					#if the gene doesn't belong to mRNA_t,CNA_t,met_t and snp_t, select the dataset that neighboring genes belong to the most.
					
					#m_score,c_score,t_score and s_score are scores that count how many neighboring genes belong to mRNA_t, CNA_t, met_t and snp_t respectly.
					m_score=0
					c_score=0
					t_score=0
					s_score=0
					m=self.gene2num[bi]
					for num,s in enumerate(flag[m]):
						if s!=0:
							adj_gene=self.num2gene[num]
							if adj_gene in mRNA_t:
								m_score+=1
							if adj_gene in CNA_t:
								c_score+=1
							if adj_gene in met_t:
								t_score+=1 
							if adj_gene in snp_t:
								s_score+=1 
								
					#select the dataset that neighboring genes belong to the most.
					total=[m_score,c_score,t_score,s_score]
					tmp=[i for i, j in enumerate(total) if j == max(total)]
					if 0 in tmp:
						data_tmp.append(self.mRNA.ix[bi,sample].values.astype('float64'))
						test_tmp.append(self.mRNA.ix[bi,test_sample].values.astype('float64'))
					if 1 in tmp:
						data_tmp.append(self.CNA.ix[bi,sample].values.astype('float64'))
						test_tmp.append(self.CNA.ix[bi,test_sample].values.astype('float64'))
					if 2 in tmp:
						data_tmp.append(self.met.ix[bi,sample].values.astype('float64'))
						test_tmp.append(self.met.ix[bi,test_sample].values.astype('float64'))
					if 3 in tmp:
						data_tmp.append(self.snp.ix[bi,sample].values.astype('float64'))
						test_tmp.append(self.snp.ix[bi,test_sample].values.astype('float64'))
			
			traindata= np.array(data_tmp).T
			testdata = np.array(test_tmp).T
			testlable = self.lable[test_sample].values.astype(np.int)
			trainlable = self.lable[sample].values.astype(np.int)
			
			rand=0
			#classifiers is the list of classifiers.
			classifiers = [
				MLPClassifier(hidden_layer_sizes=([5]), alpha = 100,max_iter=3000, random_state=rand),
				MLPClassifier(hidden_layer_sizes=([5]), alpha = 150,max_iter=3000, random_state=rand),
				MLPClassifier(hidden_layer_sizes=([5]), alpha = 200, max_iter=3000, random_state=rand),
				MLPClassifier(hidden_layer_sizes=([10]), alpha = 100, max_iter=3000, random_state=rand),		  
				MLPClassifier(hidden_layer_sizes=([10]), alpha = 150, max_iter=3000, random_state=rand),
				MLPClassifier(hidden_layer_sizes=([10]), alpha = 200, max_iter=3000, random_state=rand)
			]
			#measure area under the curve. 
			aucs =[]		 
			for name, clf in zip(names, classifiers) :	 
				pipe_lr = Pipeline([('clf', clf)])	
				probas_ = pipe_lr.fit(traindata, trainlable).predict_proba(testdata)
				fpr, tpr, thresholds = roc_curve(testlable,probas_[:,1])
				roc_auc = auc(fpr, tpr)
				classi[name].append(roc_auc)
		#calculate average of AUCs in 10 fold validation.
		total=[]
		for i in names:
			total.append(classi[i])
		total=np.array(total)
		mean=total.mean(axis=1)
		
		print(' layer=\t[5]\talpha=\t100\t10fold AUC=\t',mean[0])
		print(' layer=\t[5]\talpha=\t150\t10fold AUC=\t',mean[1])
		print(' layer=\t[5]\talpha=\t200\t10fold AUC=\t',mean[2])
		print(' layer=\t[10]\talpha=\t100\t10fold AUC=\t',mean[3])
		print(' layer=\t[10]\talpha=\t150\t10fold AUC=\t',mean[4])
		print(' layer=\t[10]\talpha=\t200\t10fold AUC=\t',mean[5])
		
   
		




if __name__=="__main__":
	main()
