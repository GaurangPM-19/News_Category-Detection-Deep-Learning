from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load the dataset
news_cleaned = pd.read_csv('uci-news-aggregator.csv')

# Load the model
model = load_model('weights.best.hdf5')

# Function to get the news link
def get_news_link(heading):
    news_link = news_cleaned[news_cleaned['TITLE'] == heading]['URL']
    return news_link.values[0] if not news_link.empty else "News link not found."

# Function to predict the category of the news
def predict_news_category(heading):
    # Preprocessing
    n_most_common_words = 1000
    max_len = 130
    tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(news_cleaned["TITLE"].values)
    sequences = tokenizer.texts_to_sequences(news_cleaned["TITLE"].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=max_len)
    print('Shape of data tensor:', data.shape)

    # Predict the category of the news
    heading = heading
    new_heading = [heading]
    seq = tokenizer.texts_to_sequences(new_heading)
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    labels = ['Business', 'Entertainment', 'Health', 'science and technology']

    a = [pred, labels[np.argmax(pred)]]
    probabilities = a[0][0]
    all_less_than_0_5 = all(prob < 0.5 for prob in probabilities)
    if all_less_than_0_5:
        a[1] = 'Other News'
        
    print("Your news category is:", a[1])
    # Get the link of the news
    news_link = get_news_link(heading)

    return a[1], news_link

@app.route('/')
def home():
    return render_template('index.html')

'''@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        headline = request.form['headline']

        # Predict the category of the news
        category, link = predict_news_category(headline)

        return render_template('prediction.html', headline=headline, category=category, link=link)

    return redirect(url_for('home'))
'''
# Your API key and Custom Search Engine ID
API_KEY = "AIzaSyC1_2wIQAc6T6A324tbu9eS7PqZOeHH6vY"
CSE_ID = "33a4502af87a14cca"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        headline = request.form['headline']

        # Predict the category of the news
        category, _ = predict_news_category(headline)

        # Perform a Google search using the Custom Search API and extract the first link
        search_query = f"{headline} news"
        api_url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key={API_KEY}&cx={CSE_ID}"
        search_results = requests.get(api_url)
        search_results_json = search_results.json()
        
        first_link = "Link not found."
        if 'items' in search_results_json:
            first_link = search_results_json['items'][0]['link']

        return render_template('prediction.html', headline=headline, category=category, link=first_link)

    return redirect(url_for('home'))
if __name__ == '__main__':
    app.run(debug=True)
