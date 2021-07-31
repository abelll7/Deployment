from django.shortcuts import render



def index(request):
    return render(request,'home.html')

def result(request):
     
    if request.method == 'POST':
        print('success')
        print(request.POST.get('Textname'))
        tr=request.POST.get('Textname')
        predict=pred(tr)
        print(predict)

        pri = { 'predict' : predict
        }

    return render(request,'result.html', pri )

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pyttsx3


resampled=pd.read_csv('C:/Deployment/Deployment/resampled.csv')
x=resampled.iloc[:,0:500]
y=resampled.iloc[:,-1]

vectorizer = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 500)
tf_idf_matrix = vectorizer.fit_transform(x,y)
#df_tfidfvect = pd.DataFrame(data = tf_idf_matrix.toarray(),columns = vectorizer.get_feature_names())
# df_tfidfvect



def pred(text):    

    a=pickle.load( open('C:/Deployment/Deployment/nlp/random1.pkl', 'rb'))
    ret = a.predict(vectorizer.transform([text]).toarray())[0]


    engine=pyttsx3.init()
    engine.setProperty('rate',125)
    
    engine.say('Your sentiment is {}'.format(ret))
    engine.runAndWait()    

    
    return ret
    

    