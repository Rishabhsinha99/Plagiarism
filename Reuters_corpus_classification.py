
from gensim.models import Word2Vec as md
from word2vec import dictry
from nltk.corpus import reuters
import numpy as np
import pandas as pd
#%%

#words_txt=gutenberg.words(['austen-emma.txt','austen-persuasion.txt','bryant-stories.txt'])
lst=reuters.fileids()
words_txt=reuters.words(fileids=lst)

string=words_txt[:]
text=' '.join(string)

data=dictry(text)

model=md(data,min_count=5,size=100,window=10,sample=1e-3)    

#%%
def transformer(X):
    aux=reuters.words(X)
    aux=[i.lower() for i in aux]
    vec_lst=[]
    for i in aux:
        try:
            vec_lst.append(model[i])
        except KeyError:
            pass

    sumlst=np.zeros(100)
    for k in vec_lst:
        sumlst=np.add(sumlst,k)
    avg=sumlst/len(vec_lst)

    return avg
#%%
X=[]
for doc in lst:
    temp=transformer(doc)
    X.append(temp)

#%%
from sklearn.model_selection import train_test_split
y=[]
for j in lst:
    y.append(reuters.categories(j)[0])
    
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=11)
    
#%%
from sklearn.ensemble import RandomForestClassifier as rfc

model=rfc(n_estimators=150)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)    

scr=model.score(X_test,y_test)
print(scr)

#%%
from sklearn.naive_bayes import GaussianNB as nb

model_nb=nb() 
model_nb.fit(X_train,y_train)

scr_nb=model_nb.score(X_test,y_test)
print(scr_nb)

#%%
from sklearn.svm import SVC

model_svm=SVC(kernel='poly')
model_svm.fit(X_train,y_train)

scr_svm=model_svm.score(X_test,y_test)
print(scr_svm)

#%%
from sklearn.neighbors import KNeighborsClassifier as knn

model_knn=knn()
model_knn.fit(X_train,y_train)

scr_knn=model_knn.score(X_test,y_test)
print(scr_knn)

#%%
arr=np.array([scr,scr_nb,scr_svm,scr_knn])
arr=arr.reshape(1,4)
prfm_scr=pd.DataFrame(arr,columns=['Random Forest','Naive Bayes','SVM','KNN'])




















