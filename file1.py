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

