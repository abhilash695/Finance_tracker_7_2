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




   def get_summary(self):
        try:
            return summarize(self.text, ratio=0.3)
        except:
            return "Summary not available due to short input."

    def named_entities(self):
        return [(ent.text, ent.label_) for ent in self.doc.ents]

    def detect_language(self):
        try:
            return self.blob.detect_language()
        except:
            return "Language detection failed."

        def extract_topics(self):
        # Simple noun chunk frequency
        noun_chunks = [chunk.text.lower() for chunk in self.doc.noun_chunks]
        top_chunks = Counter(noun_chunks).most_common(5)
        return top_chunks

    def pos_statistics(self):
        pos_counts = Counter([token.pos_ for token in self.doc])
        return dict(pos_counts)

    def analyze(self):
        print("\n=== AI ANALYSIS ===")
        print("Language Detected:", self.detect_language())
        print("Sentiment:", self.get_sentiment())
        print("\nSentence-Level Sentiment:")
        for item in self.get_sentence_level_sentiment()[:3]:
            print(f"  - {item['sentence']} => Sentiment: {item['sentiment']:.2f}")


print("\nTop Keywords:", self.get_keywords())
        print("\nNamed Entities:", self.named_entities())
        print("\nTopics Extracted (Top Noun Phrases):", self.extract_topics())
        print("\nPOS Tag Statistics:", self.pos_statistics())
        print("\nSummary:\n", self.get_summary())

def ai_demo():
    from text_input import TextLoader
    loader = TextLoader("sample.txt")
    loader.read_file()
    loader.clean_text()

    analyzer = AIAnalyzer(loader.cleaned_text)
    analyzer.analyze()

if _name_ == "_main_":
    ai_demo()





