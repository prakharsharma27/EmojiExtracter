# EmojiExtracter
It analysis your text and gives you an emoji at 70% accuracy.

Emoji Finder


Objective :-  This program will provide emoji as per the mood of the sentence.
		  By analysis using Machine Learning.

Technology Used :- Django(Python), JQuery Ajax, Javascript, HTML , ML.

Machine Learning model which is used here is Support-Vector-Machine.

What’s in File :- In Django Project there are multiple files.
		
		        Files in which Modifications have been made.

			1. Settings.py - To add Templates Folder.
			2. urls.py -  To add urls for different views.
			3. Views.py - To add different functions.
			4. models.py - To add Training model of ML.
			5. Template Folder - 
						   a. Index.html - All front end of file
								       including Javascript.
			6. Corpus.sav - File containing Bag of Words created in Training.
			7. trained.sav - File containing trained classifier object.
			8. text_emotion.csv - File on data on which ml model is trained.
		


Idea of the Project :- User will be shown a web page through html and when this page loads onload event will be called by javascript which invokes trainTest() which  makes Jquery Ajax request to train() in models.py which after data preprocessing and creating bag of words named corpus save it in corpus.sav file and then trains SVM model and store classifier object in trained.sav file all saving done using pickle.dump , then text is taken in html by texture named text . Using javascript events we are utilising onkeyup event. This gets us the value inserted by user in real time, and then it calls function printAgainkarega() . And then Jquery Ajax makes call to a textShow in view.py . Then it checks if request is made by Ajax then starts working . It loads the corpus the bag of words from corpus.sav file and classifier from trained.sav file . And add new input in corpus and then fit_transforms it and then predict value . As per the predicted value it send the emoji code in response which is then received in success tag of Ajax call in printAgainkarega. Then using JSON.stringify we made string out of objects . Then we slice that string to avoid double quotes . Which is then inserted in replaceable-text which is button. Button has an onclick function which invokes function addEmoji() which adds this button value into the string of textarea.Settings.py


TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [‘templates'],  #this is added here 
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]


Urls.py

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('textShow/', views.textShow, name='textShow'),
    path('models/',models.train,name='train'),
    # path('about',views.about,name='about')
]

Models.py
from django.db import models
import os
from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse

def train(request):
    # sourcery skip: de-morgan, extract-method, hoist-statement-from-if, remove-zero-from-range
    import numpy as np
    import pandas as pd
    import pickle
    print("lelel")
    if(os.path.getsize("./corpus.sav")>0):
        print("chal chal")
        return render(request,'index.html')
    else:
        print("heghe")
        dataset = pd.read_csv("./text_emotionNew.csv")

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
        # print(dataset)
        import re
        import nltk

        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer

        corpus = []
        for i in range(0, 5000):
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
        y=y[0:5000]
        pickle.dump(corpus,open('corpus.sav', 'wb'))
        from sklearn import svm
        # print(len(X))
        # print(len(y))
        classifier = svm.SVC()
        classifier.fit(X, y)
        print("classified")
        pickle.dump(classifier,open('trained.sav','wb'))
        return render(request,'index.html')

Views.py
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

# temp


def index(request):

    # temp = corpus
    return render(request, 'index.html', {})


def textShow(request):
    if request.is_ajax():
        nltk.download('stopwords')

        print('in')
        val = request.GET.get('val')
        print(val)
        # corpus = temp
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
        # print(len(FinalPred))
        # print(FinalPred)
        FinalPred = FinalPred[0:len(x[0])]
        # print(len(FinalPred))
        result = classifier.predict(FinalPred.reshape(-1, len(x[0])))
        val = (result[0])
        if(val=='happiness'):
            val="&#128522"
        elif(val=='sadness'):
            val="&#128532"
        # val=val.substring(1,val.length-1)
        return JsonResponse({'retype': val})

Index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Index</title>
    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
</head>
<body id="bodyid" onload="trainTest()">
<!--hello siyappas from {{name}} who's from {{place}}       &lt;!&ndash;to access parameters from views&ndash;&gt;-->
<form action="/models" method="get">
<h4>Yaha se text lekr hu about wale function mein jo hai views mein.</h4>
<textarea name="chatinput" id="text"></textarea>
<!--<button type="submit">Send</button>-->
<!--    <p id="replacable-content">{{retype}}</p>-->

</form>
<button id="replaceable-content" onclick="addEmoji()" class="col-6" >
<!--    <h2>{{retype}}</h2>-->
  </button>
<!--<script>-->
<!--    var inputBox = document.getElementById('chatinput');-->

<!--    inputBox.onkeyup = function(){-->

<!--    // document.getElementById('printchatbox').innerHTML = inputBox.value;-->
<!--}-->
<!--</script>-->
<script>
    const givenText=document.getElementById('text')
    const changable=document.getElementById('replaceable-content')
    const printAgainkarega=(val)=>{
        $.ajax({
            type:'GET',
            url:'textShow/',
            data:{
                'val':val
            },
            dataType:'json',
            success: function(data){
                // console.log(data)
                var str=`${(JSON.stringify(data['retype']))}`
                str=str.substring(1,str.length-1)
                changable.innerHTML=str
            }
            ,
            error:function (error){

                console.log(error)
            }

        })
    }
    givenText.addEventListener("keyup", e=>{

        printAgainkarega(e.target.value)
    })

    // const bodyTag=document.getElementById('bodyid')

        function trainTest() {

            $.ajax({
                type: 'GET',
                url: 'models/'
                ,
                success: function (data) {
                    // console.log(data)
                    console.log = `${JSON.stringify(data)}`
                }
                ,
                error: function (error) {

                    console.log(error)
                }

            })
        }
        function addEmoji(){
            // alert(changable.innerHTML)
            givenText.value +=changable.innerHTML


        }
</script>
</body>

</html>

