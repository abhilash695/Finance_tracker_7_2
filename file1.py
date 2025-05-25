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




def pos_tagging(self):
        return pos_tag(self.tokens)

    def sentence_tokenize(self):
        return self.sentences

    def named_entity_recognition(self):
        return ne_chunk(pos_tag(self.tokens))

    def word_frequency(self):
        filtered = self.remove_stopwords()
        return Counter(filtered).most_common(10)

    def generate_ngrams(self, n=2):
        tokens = self.remove_stopwords()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return list(ngrams)[:10]

    def sentiment_analysis(self):
        return self.sia.polarity_scores(self.text)
    def generate_wordcloud(self):
        filtered = ' '.join(self.remove_stopwords())
        wc = WordCloud(width=800, height=400, background_color='white').generate(filtered)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud")
        plt.show()

    def analyze_all(self):
        print("\n=== NLP PROCESSING ===")
        print("Original Tokens:", self.tokens[:10])
        print("Stopwords Removed:", self.remove_stopwords()[:10])
        print("Stemmed Words:", self.stem_words()[:10])
        print("Lemmatized Words:", self.lemmatize_words()[:10])
        print("POS Tags:", self.pos_tagging()[:10])
        print("Sentences:", self.sentence_tokenize()[:2])
        print("NER:", self.named_entity_recognition())
        print("Top Word Frequencies:", self.word_frequency())
        print("Bigrams:", self.generate_ngrams(2))
        print("Sentiment:", self.sentiment_analysis())
        self.generate_wordcloud()

def run_nlp_demo():
    from text_input import TextLoader
    loader = TextLoader("sample.txt")
    loader.read_file()
    loader.clean_text()

    processor = NLPProcessor(loader.cleaned_text)
    processor.analyze_all()

if _name_ == "_main_":
    run_nlp_demo()
