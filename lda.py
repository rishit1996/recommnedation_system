from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
#from gensim import corpora
#from sklearn.utils.extmath import randomized_svd
#import numpy as np
 # Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
#  Train the model on the corpus.
lda = LdaModel(common_corpus, num_topics=2,id2word=common_dictionary)

temp_file = datapath("model")
lda.save(temp_file)

#lda = LdaModel.load(datapath("model"))
other_texts = [
['computer', 'time', 'graph'],
['survey', 'response', 'eps'],
['human', 'system', 'computer']]
other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
unseen_doc = other_corpus[0]
vector = lda[unseen_doc] # get topic probability distribution for a document
lda.update(other_corpus)
vector = lda[unseen_doc]
#a=lda.print_topics(num_topics=2,num_words=3)
#U, S, V = randomized_svd(np.asarray(a), n_components=2,n_iter=5,random_state=None)
#A=U.dot(np.diag(S))

print(lda.print_topic(0))