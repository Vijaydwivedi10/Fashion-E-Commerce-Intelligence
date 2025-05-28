# nlp_utils.py

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yake
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Initialize NLP tools
analyzer = SentimentIntensityAnalyzer()
keyword_extractor = yake.KeywordExtractor(lan="en", n=1, top=5)
summarizer = LexRankSummarizer()
