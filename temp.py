import util
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans
import random
from sklearn.decomposition import NMF 
data=util.read_file('papers/papers_dataset.csv')
data['Authors']=data['Authors'].apply(lambda row : row.split(','))
data['Affiliations']=data['Affiliations'].apply(lambda row : row.split(','))
data['Author Keywords']=data['Author Keywords'].apply(lambda row : str(row).split(';'))
data['Index Keywords']=data['Index Keywords'].apply(lambda row : str(row).split(';'))

data.set_index('ID',inplace=True)

keywords=list()
for keywordsArr in data['Index Keywords']:
    for eachWord in keywordsArr:
        keywords.append(eachWord)


papers=list()
for papersArr in data.index.values:
    papers.append(papersArr)
    
    
papers=pd.DataFrame({'papers' : papers})        
keywords=pd.DataFrame({'keywords' : keywords})
papers=papers.applymap(lambda x : str(x).strip()).drop_duplicates()
keywords=keywords.applymap(lambda x : str(x).strip()).drop_duplicates()
users_keywords=util.read_file('papers/Users.csv')
users_papers_ratings=util.read_file('papers/Users.csv')

for index, row in keywords.iterrows():
    users_keywords[row[0]]=0

for index, row in papers.iterrows():
    users_papers_ratings[row[0]]=0
    

users_papers_ratings.set_index('Users',inplace=True)
users_keywords.set_index('Users',inplace=True)

for index, row in users_papers_ratings.iterrows():
    row=row.apply(lambda x : random.randrange(-1,10,4))
    users_papers_ratings.loc[index]=row

papers_indexes=list(users_papers_ratings.columns.values)

user_papers={}
for index, row in users_papers_ratings.iterrows():
    user_papers[index]=list()
for index, row in users_papers_ratings.iterrows():
    for i in range(1,len(papers_indexes)):
        if(row.iloc[i]!=-1):
            user_papers[index].append(int(papers_indexes[i]))



for key, value in user_papers.items():
    for eachPaperIdx in value:
        for eachWord in data.loc[eachPaperIdx]['Index Keywords']:
            users_keywords.loc[key][eachWord.strip()]=users_keywords.loc[key][eachWord.strip()]+1



U, S, V = randomized_svd(np.asarray(users_keywords), n_components=2,n_iter=5,random_state=None)
A=U.dot(np.diag(S))





clusters=range(2,15)
sum_of_squared_dist=[]
for k in clusters:
    km = KMeans(n_clusters=k, algorithm='auto',random_state=170)  
    km.fit_predict(A)
    sum_of_squared_dist.append(km.inertia_)
    
plt.plot(clusters, sum_of_squared_dist, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

y_pred = KMeans(n_clusters=7, algorithm='auto',random_state=170).fit_predict(A)


plt.scatter(A[:,0],A[:,1], c=y_pred)
plt.title('Users Clusters')

cluster_users=util.read_file('papers/Users.csv')
cluster_users['cluster_label']=y_pred
cluster_users.set_index('Users',inplace=True)

user_id='User-13'
cluster_no=int(cluster_users.loc[user_id])


users_list=[]
for index, row in cluster_users.iterrows():
    if(cluster_no==int(row)):
        users_list.append(str(index))

clustered_users_ratings=users_papers_ratings.loc[users_list]
clustered_user_keywords=users_keywords.loc[users_list]
clustered_users_ratings=clustered_users_ratings.applymap(lambda x : 0 if (x==-1) else x)






nmf = NMF()
W = nmf.fit_transform(clustered_users_ratings);
H = nmf.components_;
nR = np.dot(W,H)
print (nR)
