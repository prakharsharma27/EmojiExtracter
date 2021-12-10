# I have created this file - Prakhar sharma

from django.http import HttpResponse
from django.shortcuts import render
from django.core.files import File
from django.template.loader import render_to_string
from django.http import JsonResponse
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def index(request):
  return render(request, 'index.html', {})


def textShow(request):
    if request.is_ajax():
        nltk.download('stopwords')

        print('in')
        val = request.GET.get('val')
        print(val)

        corpus = pickle.load(open('corpus.sav', 'rb'))
        classifier = pickle.load(open('trained.sav', 'rb'))
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer()
        x = cv.fit_transform(corpus).toarray()
        pred = val
        review = re.sub('[^a-zA-Z]', ' ', pred)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

        FinalPred = cv.fit_transform(corpus).toarray()
        FinalPred = FinalPred[len(corpus) - 1]

        FinalPred = FinalPred[0:len(x[0])]

        result = classifier.predict(FinalPred.reshape(-1, len(x[0])))
        val = (result[0])
        if(val=='happiness'):
            val="&#128522"
        elif(val=='sadness'):
            val="&#128532"

        return JsonResponse({'retype': val})

#
# def train(request,val):
#     import numpy as np
#     import pandas as pd
#
#     dataset = pd.read_csv("./text_emotion.csv")
#
#     dataset.drop('tweet_id', axis=1, inplace=True)
#     dataset.drop('author', axis=1, inplace=True)
#
#     index_names = dataset[dataset['sentiment'] == 'empty'].index
#     dataset.drop(index_names, inplace=True)
#     index_name = dataset[dataset['sentiment'] == 'neutral'].index
#     dataset.drop(index_name, inplace=True)
#
#     dataset.reset_index(drop=True, inplace=True)
#     # print(dataset)
#     first_column = dataset["sentiment"]
#     dataset.drop('sentiment', axis=1, inplace=True)
#     dataset.insert(1, 'sentiment', first_column)
#     dataset['sentiment'].replace({'worry': 'sadness'}, inplace=True)
#     dataset['sentiment'].unique()
#     dataset['sentiment'].replace(
#         {'boredom': 'sadness', 'fun': 'happiness', 'love': 'happiness', 'enthusiasm': 'happiness',
#          'relief': 'happiness'}, inplace=True)
#     dataset['sentiment'].replace({'hate': 'angry', 'empty': 'neutral'}, inplace=True)
#     dataset['sentiment'].replace({'surprise': 'happiness'}, inplace=True)
#     print(dataset)
#     import re
#     import nltk
#     nltk.download('stopwords')
#     from nltk.corpus import stopwords
#     from nltk.stem.porter import PorterStemmer
#     corpus = []
#     for i in range(0, 5000):
#         review = re.sub('[^a-zA-Z]', ' ', dataset['content'][i])
#         review = review.lower()
#         review = review.split()
#         ps = PorterStemmer()
#         all_stopwords = stopwords.words('english')
#         all_stopwords.remove('not')
#         review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
#         review = ' '.join(review)
#         corpus.append(review)
#
#     from sklearn.feature_extraction.text import CountVectorizer
#     cv = CountVectorizer()
#     X = cv.fit_transform(corpus).toarray()
#     y = dataset.iloc[:, -1].values
#
#     from sklearn import svm
#     classifier = svm.SVC()
#     classifier.fit(X, y)
#
#     pred = val
#     review = re.sub('[^a-zA-Z]', ' ', pred)
#     review = review.lower()
#     review = review.split()
#     ps = PorterStemmer()
#     all_stopwords = stopwords.words('english')
#     all_stopwords.remove('not')
#     review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
#     review = ' '.join(review)
#     corpus.append(review)
#
#     FinalPred = cv.fit_transform(corpus).toarray()
#     result = classifier.predict(FinalPred[len(corpus) - 1].reshape(-1, 8578))
#     return(result[0])
