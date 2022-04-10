from flask import Flask,render_template,request 
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import re
import pickle


def inputPreprocessandtoken(textinput):    

    textinput = re.sub("(<.*?>)", "", textinput)
    textinput = re.sub(r'http\S+', '', textinput)
    textinput= re.sub(r"(#[\d\w\.]+)", '', textinput)
    textinput= re.sub(r"(@[\d\w\.]+)", '', textinput)
    textinput = re.sub("(\\W|\\d)", " ", textinput)
    textinput = textinput.strip()
    textinput = word_tokenize(textinput)
    porter = PorterStemmer()
    stemTextmessage = [porter.stem(word) for word in textinput]
        
    return stemTextmessage

def textPreprocessFunction(data):    
    data = re.sub("(<.*?>)", "", data)
    data = re.sub(r'http\S+', '', data)
    data= re.sub(r"(#[\d\w\.]+)", '', data)
    data= re.sub(r"(@[\d\w\.]+)", '', data)
    data = re.sub("(\\W|\\d)", " ", data)
    data = data.strip()
    data = word_tokenize(data)
    porter = PorterStemmer()
    stemming_inputdata = [porter.stem(word) for word in data]
        
    return stemming_inputdata


Model = pickle.load(open('model\suicide_model.sav', 'rb'))

model = pickle.load(open('model\emotion_model.sav', 'rb'))
app=Flask(__name__) 

@app.route('/')
def index():
    return render_template('userInput.html',data=[{'cat':'Both'},{'cat':'Emotion'},{'cat':'Suicide'}])

@app.route('/predict',methods=['GET','POST'])
def predict():
    user_input=request.form['selection']
    message = request.form['message']
    
    predictionVect = model.predict([message])[0]
    predVect1 = Model.predict([message])[0]

    resultDict1={"suicide": "Sucide", "non-suicide": "Non-Sucide"}

    resultDict = {"angry": "Angry", "disgust": "Disgust", "fear": "Fear", "happy": "Happy", "sad": "Sad ", "joy": "Joy ", "love": "Love"}

    if user_input=='Both':
        predict=[resultDict[predictionVect],resultDict1[predVect1]]
        return render_template('userInput.html',predict=predict,user_input=user_input,data=[{'cat':'Both'},{'cat':'Emotion'},{'cat':'Suicide'}])
    if user_input=='Emotion':
        predict=[resultDict[predictionVect]]
        return render_template('userInput.html',predict=predict,user_input=user_input,data=[{'cat':'Both'},{'cat':'Emotion'},{'cat':'Suicide'}])
    if user_input=='Suicide':
        predict=[resultDict1[predVect1]]
        return render_template('userInput.html',predict=predict,user_input=user_input,data=[{'cat':'Both'},{'cat':'Emotion'},{'cat':'Suicide'}])
    else:
        print(ValueError)
    
app.run(debug=True)
 