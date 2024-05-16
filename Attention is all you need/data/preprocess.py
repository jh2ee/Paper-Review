import spacy
from torchtext.data import Field, BucketIterator

# 독일어 문장을 토큰화
def tokenize_de(text):
    spacy_de = spacy.load('de_core_news_sm') # 독일어 토큰화(tokenization)
    return [token.text for token in spacy_de.tokenizer(text)]

# 영어 문장을 토큰화
def tokenize_en(text):
    spacy_en = spacy.load('en_core_web_sm') # 영어 토큰화(tokenization)
    return [token.text for token in spacy_en.tokenizer(text)]