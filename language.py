import nltk as nlp
stop_words=set(nlp.corpus.stopwords.words('english'))
r=input("Enter the answer: ")
refined=[]
for r in r.split(' '):
    if r not in stop_words:
        refined.append(r)
ls= nlp.stem.WordNetLemmatizer()
ps=nlp.stem.PorterStemmer()
refined_stem=[]
refined_lemm=[]
for a in refined:
    refined_stem.append(ls.lemmatize(a))
    refined_lemm.append(ps.stem(a))
print("\n")
print(len(refined))
print(len(refined_stem))
print(len(refined_lemm))
print(refined)
print(refined_stem)
print(refined_lemm)