import pandas as pd
import util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

data=util.read_file('papers/papers_dataset.csv')
data['Abstract']=util.text_processing(data['Abstract'])
tfidf_vectorizer_abstract=TfidfVectorizer()

input_query=["Big Data Analysis"]
input_query=pd.DataFrame({'input':input_query})
input_query=util.text_processing(input_query['input'])
input_query=input_query[0]
processing_data=list([input_query])

for value in data['Abstract']:
    processing_data.append(value)

tfidf_vectorizer=TfidfVectorizer()
tfidf_matrix_train=tfidf_vectorizer.fit_transform(processing_data)
scores=cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)

scores=scores=np.asarray(scores[:,1:]).reshape(data.shape[0],1)

data['scores']=scores
data=data.sort_values(by=['scores'],ascending=False)


for title in data['Title'][:10]:
    print(title,end='\n\n')