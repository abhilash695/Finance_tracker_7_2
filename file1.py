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


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

class NLPProcessor:
    def _init_(self, text):
        self.text = text
        self.tokens = word_tokenize(text)
        self.sentences = sent_tokenize(text)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()

    def remove_stopwords(self):
        stops = set(stopwords.words('english'))
        filtered = [word for word in self.tokens if word.lower() not in stops and word.isalnum()]
        return filtered

    def stem_words(self):
        return [self.stemmer.stem(word) for word in self.tokens if word.isalnum()]

    def lemmatize_words(self):
        return [self.lemmatizer.lemmatize(word) for word in self.tokens if word.isalnum()]



