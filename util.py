import pandas as pd
# import pymongo
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from textblob import TextBlob
from textblob import Word

def text_processing(data):
    #Transform text into lowercase
    data=data.apply(lambda x: " ".join(x.lower() for x in x.split()))
    #Removinf Puncctuation
    data=data.str.replace('[^\w\s]','')
    #Remove stopwords
    stopword_list=stopwords.words('english')
    data=data.apply(lambda x: " ".join(x for x in x.split() if x not in stopword_list))
    #Common word removal 
    freq=pd.Series(' '.join(data).split()).value_counts()[:10]
    freq=list(freq)
    data=data.apply(lambda x:" ".join(x for x in x.split() if x not in freq))
    #Rare word removal
    freq=pd.Series(' '.join(data).split()).value_counts()[-10:]
    freq=list(freq)
    data=data.apply(lambda x:" ".join(x for x in x.split() if x not in freq))
    '''
    Correct spelling
    data['Abstract']=data['Abstract'].apply(lambda x: str(TextBlob(x).correct()))
    '''
    #Stemming
    st=PorterStemmer()
    data=data.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    #Lemmatization
    data=data.apply(lambda x :" ".join([Word(word).lemmatize() for word in x.split()]))
    return data
def sentence_to_words(data):
    for rec in data['Abstract']:    
        yield()




# def insert_dataFrame(dataset, database, collection_name):
#     myclient=pymongo.MongoClient("mongodb://localhost:27017/")
#     database=myclient[database]
#     papers=database[collection_name]
#     papers.insert_many(dataset.to_dict('records'))

def read_file(file_location):
    data=pd.read_csv(file_location,encoding='ISO-8859-1').head(100)
    return data
