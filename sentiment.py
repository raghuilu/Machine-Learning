#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import nltk
rate=0
acc=[95,90,80,70,50,15,10,5]


# In[18]:


read= pd.read_csv('AFINN-111.txt',names=['word', 'score'], header=None, sep= '\t');


# In[19]:


a_d=read.to_dict(orient='index')


# In[20]:


afinn=pd.DataFrame(read)
words=afinn.iloc[:,0].values
score=afinn.iloc[:,1].values


# In[21]:


test=input("enter the sentence: ")
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))


# In[22]:


toke=nltk.word_tokenize(test)
req=[te for te in toke if not te in stopwords]
req


# In[23]:


toke=[]
for tw in req:
    if tw in words:
        ind=words.tolist().index(tw)
        toke.append(tw)
        print(str(tw) +" "+str(score[ind]))
        rate+=score[ind]
print(rate)
if(rate<0):
    print("negative")
elif(rate==0):
    print("nuetral")
else:
    print("positive")
freq=len(toke)


# In[24]:


def nueNet(rate,freq):
    raw=int(rate/freq)
    print(raw)
    if(raw==-5):
        print("Sadness: "+str(acc[1])+" fear: "+str(acc[-1])+" Anger:  "+str(acc[-1]))
    elif(raw==-4 or raw== -3):
        print("Anger: "+str(acc[3])+" Sadness: "+str(acc[-3])+ " Fear: "+str(acc[-3]))
    elif(raw==-2 or raw==-1):
        print("Fear:"+str(acc[-4])+" Sadness: "+str(acc[-4])+ " anger: 10")
    elif(raw==2 or raw==3 ):
        print("Joy: "+str(acc[2])+" surprise: "+str(acc[-2])+ " nuetral: "+ str(acc[-2]))
    elif(raw==4 or raw==5):
        print("surprise: "+str(acc[1])+" Joy: "+str(acc[-2]))
    elif(raw==0 or raw==1 or raw==0):
        print("Nuetral: "+str(acc[0])+" Joy: "+str(acc[-1]))
print("The above values are the parts to hundreds")
nueNet(rate,freq)


# In[ ]: