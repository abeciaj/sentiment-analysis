from flask import Flask, render_template, request
from transformers import pipeline
import nltk

app = Flask(__name__, template_folder='templates')

nltk.download('vader_lexicon')  # Download the lexicon for sentiment analysis

# Load the sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    # Use the transformer model for sentiment analysis
    results = sentiment_analysis(text)
    
    sentiment_scores = {
        'positive': 0.0,
        'negative': 0.0,
        'neutral': 0.0
    }
    
    for result in results:
        label = result['label']
        score = result['score']
        if label == 'POSITIVE':
            sentiment_scores['positive'] += score
        elif label == 'NEGATIVE':
            sentiment_scores['negative'] += score
        else:
            sentiment_scores['neutral'] += score
    
    return sentiment_scores

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment_scores = analyze_sentiment(text)
    return render_template('result.html', text=text, sentiment_scores=sentiment_scores)

if __name__ == '__main__':
    app.run(debug=True)
