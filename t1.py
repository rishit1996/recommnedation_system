import re
import numpy as np
import pandas as pd
from pprint import pprint
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim # don't skip this
import matplotlib.pyplot as plt


# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import util


data=util.read_file('papers/papers_dataset.csv')
data['Authors']=data['Authors'].apply(lambda row : row.split(','))
data['Affiliations']=data['Affiliations'].apply(lambda row : row.split(','))
data['Author Keywords']=data['Author Keywords'].apply(lambda row : str(row).split(';'))
data['Index Keywords']=data['Index Keywords'].apply(lambda row : str(row).split(';'))
data['Abstract']=util.text_processing(data['Abstract'])
data['Abstract']=data['Abstract'].apply(lambda x : gensim.utils.simple_preprocess(str(x), deacc=True))

#util.insert_dataFrame(data,'recommendation_system','papers')




















#
#data=util.getData(sample_size=50)
#vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
#dtm = vectorizer.fit_transform(data['Abstract'])
#U, S, V = randomized_svd(dtm, n_components=3,n_iter=5,random_state=None)
#A=U.dot(np.diag(S))
#A = Normalizer(copy=False).fit_transform(A)
#plt.figure(figsize=(15, 7)) 
#plt.scatter(A[:,0:1],A[:,1:2])
#similarity_matrix=np.asarray(np.asmatrix(A)*np.asmatrix(A).T)
#similarity_matrix=pd.DataFrame(similarity_matrix,index=data['Title'],columns=data['Title'])
#cluster = AgglomerativeClustering(n_clusters=20, affinity='euclidean', linkage='ward')  
#cluster.fit_predict(A)
#plt.figure(figsize=(15, 7))  
#plt.scatter(A[:,0],A[:,1], c=cluster.labels_, cmap='rainbow')
#
#