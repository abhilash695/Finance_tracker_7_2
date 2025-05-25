# nlp_processing.py
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict, Counter
from itertools import islice
import matplotlib.pyplot as plt
from wordcloud import WordCloud



# ai_features.py
from textblob import TextBlob
from gensim.summarization import keywords, summarize
import spacy
from collections import Counter
import os

# Load spaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class AIAnalyzer:
    def _init_(self, text):
        self.text = text
        self.blob = TextBlob(text)
        self.doc = nlp(text)

    def get_sentiment(self):
        polarity = self.blob.sentiment.polarity
        subjectivity = self.blob.sentiment.subjectivity
        return {"polarity": polarity, "subjectivity": subjectivity}

    def get_sentence_level_sentiment(self):
        return [{"sentence": str(sent), "sentiment": TextBlob(str(sent)).sentiment.polarity}
                for sent in self.blob.sentences]

    def get_keywords(self):
        try:
            return keywords(self.text, ratio=0.1, words=10).split('\n')
        except:
            return ["Not enough data to extract keywords."]




