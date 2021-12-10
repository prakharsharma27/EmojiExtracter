from django.db import models
import os
from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse


def train(request):
    import numpy as np
    import pandas as pd
    import pickle
    print("lelel")
    if (os.path.getsize("./corpus.sav") > 0):
        print("Already Trained")
        return render(request, 'index.html')
    else:
        print("Training...")
        dataset = pd.read_csv("./text_emotionNew.csv")
        dataset = dataset.iloc[25000:30000, :]
        dataset.drop('tweet_id', axis=1, inplace=True)
        dataset.drop('author', axis=1, inplace=True)

        index_names = dataset[dataset['sentiment'] == 'empty'].index
        dataset.drop(index_names, inplace=True)
        index_name = dataset[dataset['sentiment'] == 'neutral'].index
        dataset.drop(index_name, inplace=True)

        dataset.reset_index(drop=True, inplace=True)
        # print(dataset)
        first_column = dataset["sentiment"]
        dataset.drop('sentiment', axis=1, inplace=True)
        dataset.insert(1, 'sentiment', first_column)
        dataset['sentiment'].replace({'worry': 'sadness'}, inplace=True)
        dataset['sentiment'].unique()
        dataset['sentiment'].replace(
            {'boredom': 'sadness', 'fun': 'happiness', 'love': 'happiness', 'enthusiasm': 'happiness',
             'relief': 'happiness'}, inplace=True)
        dataset['sentiment'].replace({'hate': 'angry', 'empty': 'neutral'}, inplace=True)
        dataset['sentiment'].replace({'surprise': 'happiness'}, inplace=True)
        print(dataset)

        print(dataset['sentiment'].value_counts())

        import re
        import nltk

        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer

        corpus = []
        for i in range(0,3629):
            review = re.sub('[^a-zA-Z]', ' ', dataset['content'][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            corpus.append(review)

        from sklearn.feature_extraction.text import CountVectorizer

        cv = CountVectorizer()
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, -1].values
        y = y[0:3629]
        pickle.dump(corpus, open('corpus.sav', 'wb'))
        from sklearn import svm

        classifier = svm.SVC()
        classifier.fit(X, y)
        print("classified")
        pickle.dump(classifier, open('trained.sav', 'wb'))
        return render(request, 'index.html')
