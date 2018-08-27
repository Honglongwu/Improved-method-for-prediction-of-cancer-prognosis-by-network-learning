import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame
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

cancertype="BRCA"
tgene_rate=400
iteration=5
n_gene=250
limit=5



#
def preprocessing():
	def loading_data():
		mRNA = pd.read_pickle("/Data2/hoilhwon/work/Data/BioData/TCGA/mRNA/"+cancertype+"_mRNA.pkl")
		CNA=pd.read_pickle("/Data2/hoilhwon/work/Data/BioData/TCGA/CNA/"+cancertype+"_CNA.pkl")
		met=pd.read_pickle("/Data2/hoilhwon/work/Data/BioData/TCGA/Met/"+cancertype+"_Met.pkl")
		snp=pd.read_pickle("/Data2/hoilhwon/work/Data/BioData/TCGA/SNP/"+cancertype+"_SNP.pkl")
		osevent=pd.read_pickle("./"+cancertype+"_osevent.pkl")
		with open('/Data1/min/180/FIsInGene_031516_with_annotations.txt', 'r') as fop :
			edges = []
			for line in fop :
				edges.append(line.strip().split('\t'))
		return mRNA,CNA,met,snp,edges,osevent

	def intersetion_data(raw_mRNA,raw_CNA,raw_met,raw_snp,raw_edges,raw_osevent) :
		#intersection among genes in dataset
		co_gene=[x for x in raw_mRNA.index if x in raw_CNA.index]
		co_gene=[x for x in raw_met.index if x in co_gene]
		co_gene=[x for x in raw_snp.index if x in co_gene]



		#intersection between genes and FIs network
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


		co_sample=[x for x in raw_mRNA.columns if x in raw_CNA.columns]
		co_sample=[x for x in raw_met.columns if x in co_sample]
		co_sample=[x for x in raw_snp.columns if x in co_sample]

		mRNA=raw_mRNA.loc[ppi_genes,co_sample]
		CNA=raw_CNA.loc[ppi_genes,co_sample]
		met=raw_met.loc[ppi_genes,co_sample]
		snp=raw_snp.loc[ppi_genes,co_sample]
		osevent=raw_osevent.loc[co_sample]
		return mRNA,CNA,snp,met,edge_list,osevent	
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
				print('err')

		return zscored_data
	def t_test(mRNA,CNA,met,snp,num2gene,good_sam,bad_sam):
			if len(good_sam)==0:
				print('error!there is no good prognosis patient')
			if len(bad_sam)==0:
				print('error!there is no bad prognosis patient')
			n_genes=len(mRNA.index)
			genes=mRNA.index
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

			t_scores=DataFrame(t_scores,index=genes)

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
			t_scores2=DataFrame(t_scores2,index=genes)

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
			t_scores3=DataFrame(t_scores3,index=genes)
			
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
			t_scores4=DataFrame(t_scores4,index=genes)
			
			return t_scores,t_scores2,t_scores3,t_scores4
	
	print('loading data...')
	raw_mRNA,raw_CNA,raw_met,raw_snp,raw_edges,raw_osevent=loading_data()
	print('preprocessing data...')
	raw_mRNA2,raw_CNA2,raw_met2,raw_snp2,edge_list,osevent=intersetion_data(raw_mRNA,raw_CNA,raw_met,raw_snp,raw_edges,raw_osevent)
	mvalues=zscore(raw_mRNA2.values.astype('float64'))
	CNAvalues=zscore(raw_CNA2.values.astype('float64'))
	metvalues=zscore(raw_met2.values.astype('float64'))
	snpvalues=zscore(raw_snp2.values.astype('float64'))
	mRNA = DataFrame(mvalues, index=raw_mRNA2.index, columns=raw_mRNA2.columns)
	CNA = DataFrame(CNAvalues, index=raw_CNA2.index, columns=raw_CNA2.columns)
	met = DataFrame(metvalues, index=raw_met2.index, columns=raw_met2.columns)
	snp = DataFrame(snpvalues, index=raw_snp2.index, columns=raw_snp2.columns)
	
	gene2num = {}
	num2gene = {}
	for i, gene in enumerate(mRNA.index):
		gene2num[gene] = i
		num2gene[i] = gene

	print('dividing samples for 10fold validation ')
	good_sam,bad_sam=seperate_good_bad_patients(mRNA.columns,osevent)
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
	print('make_ttest_values')
	
	mRNA_ttest=[]
	CNA_ttest=[]
	met_ttest=[]
	snp_ttest=[]
	for foldnum in range(10):
		print(str(foldnum)+'fold ttest start')
		goodsam=good_train_samples[foldnum]
		badsam=bad_train_samples[foldnum]
		mRNA_ttmp,CNA_ttmp,met_ttmp,snp_ttmp=t_test(mRNA,CNA,met,snp,num2gene,goodsam,badsam)
		mRNA_ttest.append(mRNA_ttmp)
		CNA_ttest.append(CNA_ttmp)
		met_ttest.append(met_ttmp)
		snp_ttest.append(snp_ttmp)
		print(str(foldnum)+'fold ttest end')
		
	Pm=PM(mRNA,CNA,met,snp,osevent,edge_list,good_train_samples,bad_train_samples,test_samples,gene2num,num2gene,mRNA_ttest,CNA_ttest,met_ttest,snp_ttest)
	
	return Pm
	

class PM:
	def __init__(self,mRNA,CNA,met,snp,osevent,edge_list,good_train_samples,bad_train_samples,test_samples,gene2num,num2gene,mRNA_ttest,CNA_ttest,met_ttest,snp_ttest):
		
		self.mRNA = mRNA
		self.CNA=CNA
		self.met=met
		self.snp=snp
		self.osevent=osevent
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

	def reconstruct_FIs_network(self):
		reconstructed_network_10fold=[]
		gene_in_reconstructed_network_10fold=[]
		for foldnum in range(10):
			selected_by_ttest=set()
			reconstructed_network=[]
			gene_in_reconstructed_network=set()
			good_sam=self.good_train_samples[foldnum]
			bad_sam=self.bad_train_samples[foldnum]
			mRNA_t_sort=self.mRNA_ttest[foldnum].sort_values(by=0,ascending=False)
			CNA_t_sort=self.CNA_ttest[foldnum].sort_values(by=0,ascending=False)
			met_t_sort=self.met_ttest[foldnum].sort_values(by=0,ascending=False)
			snp_t_sort=self.snp_ttest[foldnum].sort_values(by=0,ascending=False)
			
			selected_by_ttest.update(mRNA_t_sort.index[:tgene_rate])
			selected_by_ttest.update(CNA_t_sort.index[:tgene_rate])
			selected_by_ttest.update(met_t_sort.index[:tgene_rate])
			selected_by_ttest.update(snp_t_sort.index[:tgene_rate])

			for edge in self.edge_list:
				if edge[0] in selected_by_ttest or edge[1] in selected_by_ttest:
					reconstructed_network.append(edge)
					gene_in_reconstructed_network.update(edge)
			reconstructed_network_10fold.append(reconstructed_network)
			gene_in_reconstructed_network_10fold.append(gene_in_reconstructed_network)
		return reconstructed_network_10fold,gene_in_reconstructed_network_10fold
	def mk_data_for_GANs(self,networkgene,foldnum):
		trainsample=np.hstack((self.good_train_samples[foldnum],self.bad_train_samples[foldnum]))
		random.seed(0) 
		random.shuffle(trainsample)
		result_tmp=[]
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
	def Learning_FIsnetwork_GANs(self,process_number,edge_list,data_for_GANs,foldnum,output):
		def make_adjacencyMatrix_for_GANs(n_genes,edge_list):
			matrix = np.zeros([n_genes,n_genes], dtype=np.float32)
			for edge in edge_list:
				x = gene2num_forGANs[edge[0]]
				y = gene2num_forGANs[edge[1]]
				matrix[x][y] = matrix[y][x] = 1.
			return matrix
		def prepare(adjacency_matrix,n_input,n_hidden,n_noise,stddev):
			reconstucted_network_adjacency_matrix = tf.constant(adjacency_matrix)
			X = tf.placeholder(tf.float32, [None, n_input])

			Z = tf.placeholder(tf.float32, [None, n_noise])

			G_W = tf.Variable(tf.random_normal([n_noise, n_genes], stddev=0.01))

			D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))

			D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
			return reconstucted_network_adjacency_matrix,X,Z,G_W,D_W1,D_W2
		
		def generator(G_W,reconstucted_network_adjacency_matrix,noise_z):
			output = tf.nn.relu(tf.matmul(noise_z, reconstucted_network_adjacency_matrix*(G_W*tf.transpose(G_W))))	   
			return output

		def discriminator(inputs,D_W1,D_W2):
			hidden = tf.nn.relu(tf.matmul(inputs, D_W1))
			output = tf.nn.sigmoid(tf.matmul(hidden, D_W2))
			return output

		def get_noise(batch_size, n_noise):
			return np.random.normal(size=(batch_size, n_noise))
			
		print('process number : ',process_number,'fold number :',foldnum)
		
		total_gene=[]
		for i in edge_list:
			total_gene.append(i[0])
			total_gene.append(i[1])
		total_gene=set(total_gene)
		
		gene2num_forGANs = {}
		num2gene_forGANs = {}
		for i, gene in enumerate(total_gene):
			gene2num_forGANs[gene] = i
			num2gene_forGANs[i] = gene	
		
	  
		n_genes = len(total_gene)
		data_for_GANs=np.array(data_for_GANs)
		data_for_GANs = data_for_GANs.T
	  
		adjacency_matrix = make_adjacencyMatrix_for_GANs(n_genes,edge_list)

		tf.set_random_seed(process_number)
		batch_size = 1
		learning_rate = 0.0002

		
		# GA
		

		reconstucted_network_adjacency_matrix,X,Z,G_W,D_W1,D_W2=prepare(adjacency_matrix,n_genes,256,n_genes,0.01)

		G = generator(G_W,reconstucted_network_adjacency_matrix,Z)

		D_gene = discriminator(G,D_W1,D_W2)

		D_real = discriminator(X,D_W1,D_W2)

		loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))

		loss_G = tf.reduce_mean(tf.log(D_gene))

		D_var_list = [D_W1, D_W2]
		G_var_list = [G_W]


		train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
		train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

		n_iter = data_for_GANs.shape[0]
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		loss_val_D, loss_val_G = 0, 0
		loss_val_D_list = []
		loss_val_G_list = []
		loss_val_D_list_tmp=0
		loss_val_G_list_tmp=0
		for epoch in range(100):
			for i in range(n_iter):
				batch_xs = data_for_GANs[i].reshape(1,-1)
				noise = get_noise(1, n_genes)
	  
				_, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
				_, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})
				loss_val_D_list.append(loss_val_D)
				loss_val_G_list.append(loss_val_G)
				if i % 10== 0 :		
					loss_val_D_list_tmp=np.mean(loss_val_D_list)
					loss_val_G_list_tmp=np.mean(loss_val_G_list)
					loss_val_D_list = []
					loss_val_G_list = []
					
			if loss_val_D_list_tmp <= -0.6 and loss_val_D_list_tmp >= -0.8 and loss_val_G_list_tmp <= -0.6 and loss_val_G_list_tmp >= -0.8:
				print('#################### converge ####################','Epoch:', '%04d' % epoch,'n_iter :', '%04d' % n_iter,'D_loss : ','%04d' % loss_val_D_list_tmp,' G_loss','%04d' % loss_val_G_list_tmp)
				break
		print('done!')
		network = sess.run(reconstucted_network_adjacency_matrix*(G_W*tf.transpose(G_W)))
		result = []
		for n in range(0, len(network)):
			for m in range(n+1, len(network[n])):
				val = network[n][m]
				if val != 0 :
					result.append([num2gene_forGANs[n], num2gene_forGANs[m], val])
	  
		output.put(result)
	def pagerank(self,weight):
	
		

		damping_factor=0.7
		threshold=0.005
		## 2. adjacency matrix
		chong_gene=[]
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

	  ## 4. PageRank
		def perform_pagerank(matrix, d, threshold):
			n_genes = matrix.shape[0]
			score = np.ones(n_genes) / n_genes
			tmp = np.ones(n_genes) / n_genes

			for i in range(100):
				score = (1.-d)*(np.ones(n_genes) / n_genes) + d*(matrix.dot(score))
				deviation = np.abs(score-tmp).max()
				if deviation < threshold:
					print("##############pagerank converge##############")
					break
				else:
					tmp = deepcopy(score)
			return score

		score = perform_pagerank(result_mat, damping_factor, threshold)
		result=DataFrame(score)
		result_gene=result.sort_values(by=0,ascending=False).index
		real_result=[]
		for x in result_gene:  
			real_result.append(self.num2gene[x])
			if len(real_result)==n_gene:
				break
		return real_result
	def select_Feature(self,foldnum,gene_in_reconstructed_FIs,reconstructed_FIs):
		print(str(foldnum)+'fold select_Feature')
		data_for_GANs=pm.mk_data_for_GANs(gene_in_reconstructed_FIs,foldnum)
		
		score=np.zeros((len(self.mRNA.index)))
		output1 = Queue(); output2 = Queue(); output3 = Queue();output4 = Queue();output5 = Queue();
	
		process_list = []
		Output = [output1, output2, output3,output4,output5]
		for process_number in range(5) :
			process_list.append(Process(target=pm.Learning_FIsnetwork_GANs, args=(process_number, reconstructed_FIs,data_for_GANs,foldnum, Output[process_number])))
		print("start process")

		for n,p in enumerate(process_list) :
			print(("process_list %d call")%(n))
			p.start()	 

		

		result_GANs=[]
		result_GANs.append(output1.get());
		result_GANs.append(output2.get());
		result_GANs.append(output3.get());
		result_GANs.append(output4.get());
		result_GANs.append(output5.get());
		
		
		print("close process")
		for process in process_list :
			process.join()
		for i in range(iteration):
			pagerank_genes=pm.pagerank(result_GANs[i])
			for k in pagerank_genes:
				score[self.gene2num[k]]=score[self.gene2num[k]]+1
		biomarker=[]
		for i,j in zip(score,self.mRNA.index):
			if i >=limit:
				biomarker.append(j)
		return biomarker
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
	def auc(self,reconstructed_FIs,biomarkerperfold) :
		

		names = [n for n in range(6)]
		classi={}
		for i in names:
			classi[i]=list()
		for foldnum in range(10):
			sample=np.hstack((self.good_train_samples[foldnum],self.bad_train_samples[foldnum]))
			test_sample=self.test_samples[foldnum]
			random.shuffle(sample)
			edge_list=reconstructed_FIs[foldnum]
			edge_list=np.array(edge_list)
			genes=set()
			genes.update(edge_list[:,0])
			genes.update(edge_list[:,1])


			
			selected_by_ttest=set()
			mRNA_ttest=self.mRNA_ttest[foldnum].sort_values(by=0,ascending=False)
			CNA_ttest=self.CNA_ttest[foldnum].sort_values(by=0,ascending=False)
			met_ttest=self.met_ttest[foldnum].sort_values(by=0,ascending=False)
			snp_ttest=self.snp_ttest[foldnum].sort_values(by=0,ascending=False)

			mRNA_t=mRNA_ttest.index[:tgene_rate]
			CNA_t=CNA_ttest.index[:tgene_rate]
			met_t=met_ttest.index[:tgene_rate]
			snp_t=snp_ttest.index[:tgene_rate]
			selected_by_ttest.update(mRNA_t)
			selected_by_ttest.update(CNA_t)
			selected_by_ttest.update(met_t)
			selected_by_ttest.update(snp_t)
			data_tmp=[]
			test_tmp=[]
			flag=self.make_adjacencyMatrix(edge_list)
			biomarker=biomarkerperfold[foldnum]
			#print(len(biomarker))
			for bi in biomarker:
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
			testosevent = self.osevent[test_sample].values.astype(np.int)
			trainosevent = self.osevent[sample].values.astype(np.int)
			rand=0
			
			classifiers = [
				MLPClassifier(hidden_layer_sizes=([5]), alpha = 100,max_iter=3000, random_state=rand),
				MLPClassifier(hidden_layer_sizes=([5]), alpha = 150,max_iter=3000, random_state=rand),
				MLPClassifier(hidden_layer_sizes=([10]), alpha = 100, max_iter=3000, random_state=rand),
				MLPClassifier(hidden_layer_sizes=([10]), alpha = 150, max_iter=3000, random_state=rand),		  
				MLPClassifier(hidden_layer_sizes=([5,5,5]), alpha = 100, max_iter=3000, random_state=rand),
				MLPClassifier(hidden_layer_sizes=([5,5,5]), alpha = 150, max_iter=3000, random_state=rand)
			]
			aucs =[]		 
			for name, clf in zip(names, classifiers) :	 
				pipe_lr = Pipeline([('clf', clf)])	
				probas_ = pipe_lr.fit(traindata, trainosevent).predict_proba(testdata)
				fpr, tpr, thresholds = roc_curve(testosevent,probas_[:,1])
				roc_auc = auc(fpr, tpr)
				classi[name].append(roc_auc)
		total=[]
		for i in names:
			total.append(classi[i])
		total=np.array(total)
		mean=total.mean(axis=1)
		for n,result_mean in enumerate(mean):
			print(classifiers[n],result_mean)



	
def seperate_good_bad_patients(sampleList,osevent):
	good_samples=[]
	bad_samples=[]
	for i,j in zip(sampleList,osevent[sampleList]):
		if j=='0':
			good_samples.append(i)
		elif j=='1':
			bad_samples.append(i)
		else:
			print('error!lable can be only 0 or 1')
	return good_samples,bad_samples

#main
#args = parse_arguments()
pm=preprocessing() 

reconstructed_FIs_perfold,gene_in_reconstructed_FIs_perfold=pm.reconstruct_FIs_network()
biomarker_perfold=[]
for foldnum in range(10):
	biomarker=pm.select_Feature(foldnum,gene_in_reconstructed_FIs_perfold[foldnum],reconstructed_FIs_perfold[foldnum])
	biomarker_perfold.append(biomarker)

pm.auc(reconstructed_FIs_perfold,biomarker_perfold)
