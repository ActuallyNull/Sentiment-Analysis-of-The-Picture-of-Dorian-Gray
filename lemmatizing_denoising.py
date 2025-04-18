import nltk
from nltk.corpus import twitter_samples,stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from tokenizing import chap_token
import string

def lemmatize_chapter(chap_token):
    lemmatizer = WordNetLemmatizer() # initializing the lemmatizer
    lemmatized_sentence = []
    for word, tag in pos_tag(chap_token):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

lemm_chapter = lemmatize_chapter(chap_token)
print(lemm_chapter[0:20])

# denoising
stop_words = stopwords.words('english') # words/punctionation that are not useful for analysis (e.g. "the", "a", "an", "is", "are", "and", "or", etc.)

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'): # normalises a word withing the context of vocabulary and through morphological analysis to produce a lemma
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and (token == "“") or (token == "”") or (token == "’"):
            continue
        elif len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())

    return cleaned_tokens

denoised_chapter = remove_noise(lemm_chapter, stop_words)
#print(denoised_chapter[0:20])

