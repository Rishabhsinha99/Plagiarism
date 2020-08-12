import nltk
from nltk import re
from nltk import PorterStemmer
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
ps=PorterStemmer()
#%%
# function
def stemming(string,top):
    string_rep=re.sub('[^a-zA-Z]',' ',string)
    string_rep=string_rep.lower()
    new_vocab=word_tokenize(string_rep)
    #inter_stem=[ps.stem(l) for l in new_vocab]
    vocab_stem=[ps.stem(w) for w in new_vocab if w not in set(stopwords.words('english'))]
    dictry_stem=set(vocab_stem)
    data=' '.join(vocab_stem)
    arr=[]
    c=0
    t=0
    for i in dictry_stem:
        c=len(re.findall(i,data))
        t=t+c
        arr.append([i,c])
        #print(i,c)
    aux=[]
    for q,w in arr:
        if w>1:
            aux.append([q,w])      
    
    aux.sort(key=lambda i:i[1],reverse=True)        
    aux=[w for w in aux if w[1]>1]    
    
    x=[] 
    y=[]
    for wrd,co in aux:
        x.append(wrd)
        y.append(co)
        
    print("total number of words:",t,"\t set of words:",len(set(dictry_stem)))
    #print("Top ",top," words occurring more than twice: ",aux)
    print("%age of total length these word account for:", (sum(y)/t)*100)
    #plt.bar(x,y,color='red',alpha=0.8)
    #plt.xlabel('Word')
    #plt.ylabel('Frequency')
    #plt.title('Words occurring more than twice')
    return(x,string_rep)
#%%
#n-gram(Bigram) model
def ngram(corpus):
    new_vocab=word_tokenize(corpus)
    fd=FreqDist(new_vocab)
    words = [w for w in fd.keys()]
    most=[]
    for wr in words:
        most.append([wr,fd[wr]])
    most=sorted(most,key=lambda i:i[1],reverse=True)
    most=[a for a in most if a[0] not in set(stopwords.words('english')) ]
    most=most[:10]
  
    x=[]
    for b,c in most:
        x.append(b)
    
    new=[]
    for i in x:
        new.append([i,[n for n,val in enumerate(new_vocab) if val==i]])
    y=[]
    for j,k in new:
        y.append(k)

#bi-gram model:    
    master_prev=[]
    for z in y:
        for p in z:
            master_prev.append([(new_vocab[p-1]),(new_vocab[p])])        
    try:
       master_next=[]
       for z in y:
           for p in z:
               master_next.append([(new_vocab[p]),(new_vocab[p+1])])        
    except IndexError as error:
        pass
    master=master_next + master_prev
    dictionary=[]
    for ke in master:
        if ke not in dictionary:
            dictionary.append(ke)

    bigram=[]
    c=0
    for word in dictionary:
        for sec in master:
            if word==sec:
                c=c+1
        bigram.append([word,c])   
        c=0    
    bigram=sorted(bigram,key=lambda s:s[1],reverse=True) 
    bigram=[w for w in bigram if w[1]>1]
    
#tri-gram model    
    tri_prev=[]
    for z in y:
        for p in z:
            tri_prev.append([(new_vocab[p-2]),(new_vocab[p-1]),(new_vocab[p])])        
   
    try:
        tri_next=[]
        for z in y:
            for p in z:
                tri_next.append([(new_vocab[p]),(new_vocab[p+1]),(new_vocab[p+2])])       
    except IndexError as error:
        pass           

    tri=tri_next + tri_prev
    dictionary_=[]
    for lk in tri:
        if lk not in dictionary_:
            dictionary_.append(lk)

    trigram=[]
    ct=0
    for word in dictionary_:
        for sec in tri:
            if word==sec:
                ct=ct+1
        trigram.append([word,ct])   
        ct=0
                
    trigram=sorted(trigram,key=lambda s:s[1],reverse=True)    
    trigram=trigram[:10]
    
    return(trigram,bigram)        

#%%
#TF-IDF Vectorizer
def TFIDF(data):
    
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk import re
    import matplotlib.pyplot as plt
    data=re.sub('[^a-zA-Z]',' ',data)
    # common=[set(stopwords.words('english'))]
    # data=re.sub(common,' ',data)
    vectorizer = TfidfVectorizer()
    #data=['Today was a vergy productive day. I learned how to access the sub-indices of list and use them']
    data=[data]
    X = vectorizer.fit_transform(data)
    dense = X.todense()
    dense=np.swapaxes(dense,0,1)
    dense=dense.tolist()
    grp_x=[]
    for i in dense:
        grp_x.append(i[0])
    feature_names = vectorizer.get_feature_names()

    grp_x=np.asarray(grp_x)
    arg=len(grp_x)
    grp_x=grp_x.reshape((arg,1))
    feature_names=np.asarray(feature_names)  
    feature_names=feature_names.reshape((arg,1))
    matrix=np.concatenate((feature_names,grp_x),axis=1)
    matrix=matrix.tolist()
    matrix=sorted(matrix,key=lambda i:i[1],reverse=True)
    fin=[]
    dictry_tfidf=[]
    for j in matrix:
        if j[0] not in set(stopwords.words('english')):
            fin.append([j[0],j[1]])
            dictry_tfidf.append(j[0])    
    top=fin[:10]
    last=fin[-11:-1]
    tfidf=top+last
    tfidf=sorted(tfidf,key=lambda j:j[1],reverse=True)

   
    return(tfidf)

#%%
#gensim library
def genlib(docx,top):    
    from nltk.corpus import stopwords 
    from gensim import corpora
    import matplotlib.pyplot as plt

    docx=docx.lower()
    docx=[docx]

    texts = [[text for text in doc.split() if text not in set(stopwords.words('english'))] for doc in docx]  
    dictionary = corpora.Dictionary(texts)
    arr=[dictionary.doc2bow(doc, allow_update=True) for doc in texts]
    word_counts = [[(dictionary[id], count) for id, count in line] for line in arr]
    word_counts=[item for item in word_counts[0]]
    word_counts=sorted(word_counts,key=lambda i:i[1],reverse=True)
    word_counts=word_counts[:top]

    x=[] 
    y=[]
    for wrd,co in word_counts:
        x.append(wrd)
        y.append(co)

    plt.bar(x,y,color= 'blue')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Words occurring more than twice')
    
    return(word_counts)

#%%    
"""text classification and text summarization----> topics for paper"""
""" is a problem relevant, ur solution, does solution work---->achievable"""
"""come up with insightful problems----->rare"""
#%%
#gensim sumarizer
def summary(stry):
    from gensim.summarization.summarizer import summarize
    summary=summarize(stry,word_count=100)
    return summary

#%%
# clean text for plagiarism
def clean(string_rep):
    string_rep=re.sub('[^a-zA-Z .]',' ',string_rep)
    string_rep=string_rep.lower()
    new_vocab=word_tokenize(string_rep)
    new_vocab=[w for w in new_vocab if w not in set(stopwords.words('english'))]
    data=' '.join(new_vocab)
    
    return data    

#%%
# cosine similarity calculation
def cossim(doc1,doc2):
    from sklearn.metrics.pairwise import cosine_similarity as cs
    from sklearn.feature_extraction.text import CountVectorizer as cv

    x=[doc1,doc2]
    vectorizer=cv().fit_transform(x)
    vectors=vectorizer.toarray()    
    
    a=vectors[0].reshape(1,-1)
    b=vectors[1].reshape(1,-1)    
    
    similarity_score=cs(a,b)        
    
    return similarity_score

    

#%%
# plagiarism check for exact copy
def plgcpy(inst_1,inst_2):

    summary_1=summary(inst_1)
    summary_1_words,summary_1_vocab=stemming(summary_1,10)
    summary_1_tfidf=TFIDF(summary_1)
    words_1,vocab_1=stemming(inst_1,10)        
    trigram_1,bigram_1=ngram(vocab_1)
    
    summary_2=summary(inst_2)
    summary_2_words,summary_2_vocab=stemming(summary_2,10)
    summary_2_tfidf=TFIDF(summary_2)
    words_2,vocab_2=stemming(inst_2,10)        
    trigram_2,bigram_2=ngram(vocab_2)
    
    summary_words_match=np.in1d(summary_1_words,summary_2_words)
    num_match_summ= (summary_words_match==True).sum()
    doc_words_match=np.in1d(words_1,words_2)
    num_match_doc= (doc_words_match==True).sum()
    
    aux_bigram_1=[]
    for i in bigram_1:
        aux_bigram_1.append(i[0])
    aux_bigram_2=[]
    for j in bigram_2:
        aux_bigram_2.append(j[0])
    bigram_match=np.in1d(aux_bigram_1,aux_bigram_2)
    num_match_bigram=(bigram_match==True).sum()
    
    """aux_tfidf_1=[]
    for i in summary_1_tfidf:
        aux_tfidf_1.append(i[0])
    aux_tfidf_2=[]
    for j in summary_2_tfidf:
        aux_tfidf_2.append(j[0])
    tfidf_match=np.in1d(aux_tfidf_1,aux_tfidf_2)
    num_match_tfidf=(tfidf_match==True).sum()"""
    
    clean_summary_1=clean(summary_1)
    clean_summary_2=clean(summary_2)
    
    score=cossim(clean_summary_1,clean_summary_2)
    score=score[0][0]
    
    #lst=[num_match_bigram,num_match_doc,num_match_summ,score]

    if (num_match_summ>1 and num_match_doc>11 and num_match_bigram>7 and score>0.35) :
            print('\nPlagiarised')
            return 'Plagiarised'
    else:
            print('\nOriginal')
            return 'Original'
    
















