from flask import Flask, render_template, request
app = Flask(__name__)
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model.h5')
import json
import random
#intents = json.loads(open('intents.json').read())
intents=json.loads(open('intents.json',encoding="utf8").read())
Vocabs = pickle.load(open('Vocabs.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    S_word = nltk.word_tokenize(sentence) #split words in array
    S_word = [lemmatizer.lemmatize(word.lower()) for word in S_word] # stemming sentence
    return S_word

def bag_of_words(sentence, Vocabs, show_details=True):
    
    S_word = clean_up_sentence(sentence)
    bag = [0]*len(Vocabs)  
    for s in S_word:
        for i,word in enumerate(Vocabs):
            if word == s: 
                bag[i] = 1 # give 1 if current word £ vocabs position
                if show_details:
                    print ("found in bag: %s" % Vocabs)
    return(np.array(bag)) # result = list of 0 and 1 for each word that exist in sentence of user

def predict_class(sentence):
    p = bag_of_words(sentence, Vocabs,show_details=False)
    res = model.predict(np.array([p]))[0] #prediction
    ERROR_THRESHOLD = 0.25
    probs = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    probs.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for res in probs:
        return_list.append({"intent": classes[res[0]], "probability": str(res[1])}) #high probability
    return return_list
@app.route("/")
def home():
    return render_template("index.html")
 
@app.route("/get")
def get_bot_response():
    msg = request.args.get('msg')

    if msg != '':
       
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        
    return str(getResponse(ints,intents))
	
	
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result







if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int("5000"),debug=True)
