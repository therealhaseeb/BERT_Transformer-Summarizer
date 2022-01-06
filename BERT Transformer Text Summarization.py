from transformers import pipeline
from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok



ARTICLE = """Email classification using bert embeddings and classifier, also classify the emails
using svm, logistic regression, decision tree. Email classification using LSTM model,
tuning the hyperparameters of the model, also implemented several other scripts on email
classification dataset (dbscan.py, lda.py). Model metrics save completed, vm environment setup
will continue tomorrow as more problems occur in vm, output metrics of bert and lstm
are having problem because they are not 1d arrays, looking for workaround. Making Algorithms
presentations and waiting for Akash to give more autoML algorithms to implement.
Implementing Pytorch Sentiment Analysis, 2/6 done. Implementing Pytorch Sentiment Analysis, done 4
algorithms out of 6.
# """

ARTICLE = """Worked on predicting credit risk modelling and doing exploratory data analysis on its dataset.
 Worked on Insurance Claim Prediction, predicting the medical cost billed by medical insurance, a regression problem.
 Implemented Health Insurance Claim Prediction, predicting whether a person will take up health insurance bu seeing his other data of previous insurance and demographics.
 Worked on Reamaining Useful Life Prediction on NASA Jet Engine Dataset, prediction the useful life remaining of Jet Engines.
 Worked on Car Insurance Claim Dataset to generate some insights about car insurance claims and see what factors will make customers more likely to be repeat offenders."""

app = Flask(__name__)
# run_with_ngrok(app)


@app.route('/hi')
def index():
    return 'Welcome to BERT Text Summarization API'


@app.route('/', methods=['GET', 'POST'])
def summary():
    if request.method == 'GET':
        return render_template('mytextbox.html')

    if request.method == 'POST':
        text = request.form['text']
        print('INPUT- ', text)
        summarizer = pipeline("summarization")
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]
        print('OUTPUT- ', summary['summary_text'])
        return summary['summary_text']

import os
if __name__ == '__main__':
    app.run(port=4444)