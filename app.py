from flask import Flask,render_template,request 
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import re
import pickle

def preprocess_and_tokenize(data):    

    #remove html markup
    data = re.sub("(<.*?>)", "", data)

    #remove urls
    data = re.sub(r'http\S+', '', data)
    
    #remove hashtags and @names
    data= re.sub(r"(#[\d\w\.]+)", '', data)
    data= re.sub(r"(@[\d\w\.]+)", '', data)

    #remove punctuation and non-ascii digits
    data = re.sub("(\\W|\\d)", " ", data)
    
    #remove whitespace
    data = data.strip()
    
    # tokenization with nltk
    data = word_tokenize(data)
    
    # stemming with nltk
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]
        
    return stem_data

model = pickle.load(open('model\emotion_model.sav', 'rb')) #load the model

app=Flask(__name__) #application

@app.route('/')
def index():
    return render_template('userInput.html')

@app.route('/predict',methods=['POST'])
def predict():

    message = request.form['message']

    predictionVect = model.predict([message])[0]

    resultDict = {"angry": "Angry &#128545;", "disgust": "Disgust &#128546;", "fear": "Fear &#128547;", "happy": "Happy &#128548;", "sad": "Sad &#128549;", "joy": "Joy &#128563;"}

    return render_template('userInput.html',prediction=resultDict[predictionVect])

app.run(debug=True)
