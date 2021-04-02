import pandas as pd
import numpy as np
import math
import re, string
from sentence_transformers import SentenceTransformer
import time
from sklearn.metrics.pairwise import cosine_similarity


def read_query(path_to_query):

	file = open(path_to_query, 'r')

	text = file.read()

	file.close()

	titles = re.findall('<title> (.*) </title>', text)

	return titles

#From https://towardsdatascience.com/calculating-document-similarities-using-bert-and-other-models-b2c1a29c9630
def most_similar(doc_id,similarity_matrix):
	similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]

	for i,ix in enumerate(similar_ix):
		if ix==doc_id:
			continue
		if i > 100:
			break;
		if ix < 49:
			continue

		print(str(doc_id+1)+' Q0 '+str(common_texts[ix-49][0])+' '+str(i)+' '+str(round(pairwise_similarities[doc_id][ix],3))+' myRun')

if __name__ == '__main__':
	DATA_PATH = 'data/Trec_microblog11.txt'
	STOPWORD_PATH = 'data/stopwords.txt'
	QUERY_PATH = 'data/topics_MB1-49.txt'

	stopwords = np.loadtxt(STOPWORD_PATH, dtype=str)	

	data = pd.read_csv(DATA_PATH, sep="	",header = None)

	queries = read_query(QUERY_PATH)

	common_texts = [(row[0],row[1]) for i,row in data.iterrows()]

	sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

	tic = time.perf_counter()

	document_embeddings = sbert_model.encode(queries+[v for i,v in common_texts])

	toc = time.perf_counter()
	print(f"Encoding in {toc - tic:0.4f} seconds")

	pairwise_similarities=cosine_similarity(document_embeddings)

	for i in range(len(queries)):
		most_similar(i,pairwise_similarities)







