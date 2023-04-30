import pickle
from flask import Flask, request, jsonify
import string
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

# Creating a function to process the data
def text_process(text):
    pattern = r'https?://\S+|www\.\S+'
    text = re.sub(pattern, '', text)
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'ð', '', text)
    text = re.sub(r'rt', '', text)

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

with open('ridge1.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def predict():
    text=request.args.get('text')
    s = pd.Series([text])
    y_pred = model.predict(s)
    print(y_pred)
    if y_pred == "Hate_Speech":
        output = {"result": "This is a hate speech."}
    elif y_pred == "Offensive_Speech":
        output = {"result": "This is an offensive speech."}
    else:
        output = {"result": "This is a safe speech."}

    return jsonify(output)


if __name__ == '__main__':
    app.run()