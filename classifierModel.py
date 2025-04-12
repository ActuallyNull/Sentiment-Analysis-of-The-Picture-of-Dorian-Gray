import nltk
from nltk.corpus import twitter_samples
from nltk import NaiveBayesClassifier
import random
import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from tokenizing import chap_token
import string

#More stuff for sentiment analysis
""" nltk.download('punkt') #Pretrained model to tokenize words
nltk.download('wordnet') #Lexical database to help determine base word
nltk.download('averaged_perceptron_tagger') #Used to determine context of word in sentence
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab') """

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

#Get twitter_samples
#nltk.download('twitter_samples') #30000 tweets. 5000 positive, 5000 negative. Rest are neutral.
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')

#Clean up of all tweets
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

#Get tweets ready for modeling
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
        
positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive") 
                    for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative") 
                    for tweet_dict in negative_tokens_for_model]

#Combine positive and negative tweets
dataset = positive_dataset + negative_dataset

#Shuffle generated dataset
random.shuffle(dataset)

#Do an 80:20 split on the dataset. 80 on the testing set and 20 on the training set
train_data = dataset[:2000]
test_data  = dataset[8000:]

classifier = NaiveBayesClassifier.train(train_data)

""" print("Accuracy is:", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(10)) """

""" sentence = "I don’t want you to meet him."

custom_tokens = remove_noise(word_tokenize(sentence))
print(custom_tokens)
print(classifier.classify(dict([token, True] for token in custom_tokens))) """