from lemmatizing_denoising import denoised_chapter
from classifierModel import classifier
from chapterCategorization import split_book_into_chapters, get_book_chapters
from tokenizing import tokenize_chapter_contents
from lemmatizing_denoising import lemmatize_chapter, remove_noise
import os
import pandas as pd
from nltk.corpus import stopwords
import nltk
#More stuff for sentiment analysis
""" nltk.download('punkt') #Pretrained model to tokenize words
nltk.download('wordnet') #Lexical database to help determine base word
nltk.download('averaged_perceptron_tagger') #Used to determine context of word in sentence
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.download('vader_lexicon') """
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex

def get_chapters_for_model(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        yield dict([tokens, True] for token in tokens)

tokens_for_model = get_chapters_for_model(denoised_chapter)
""" print(denoised_chapter[0])
print('This is the dictionary for the tokenized and denoised words in Chapter 1 from:')
print("The Portrait of Dorian Gray")

print(list(tokens_for_model)[:15])

print(classifier.classify(dict([token, True] for token in tokens_for_model))) """

def book_sentiment_wrapper(book_dir, book_id,stop_words):
    
    #Start by splitting book into chapters and saving them into textfiles
    chapter_list = split_book_into_chapters(book_dir, book_id)
    num_chapters = len(chapter_list)
    
    #Initialize sentiment array
    book_sentiment = []
    cleaned_chaps  = []
    chapter_num = 0
    for chapter_num in range(num_chapters):
        #print('Processing Chapter ', chapter_num)
        chapter_contents = get_book_chapters(chapter_list, chapter_num)
        chap_token       = tokenize_chapter_contents(chapter_contents)
        lemmatized_chap  = lemmatize_chapter(chap_token)
        #print(lemmatized_chap[:10])
        denoised_chapter = remove_noise(lemmatized_chap, stop_words)
        cleaned_chaps.append(denoised_chapter)
        #print(denoised_chapter[:10])
        tokens_for_model = get_chapters_for_model(denoised_chapter)
        list(tokens_for_model)
        #print(tokens_for_model)
        chap_sentiment   = classifier.classify(dict([token, True] for token in tokens_for_model))
        
        #print('Chapter ', chapter_num, ' Sentiment is ', chap_sentiment)
        book_sentiment.append(chap_sentiment)
        
        chapter_num += 1
    
    print('Book Completed!')
    return book_sentiment, cleaned_chaps

book_id = '\\174.txt'           #What book are we processing?
book_dir = 'books' #Where is the book located?
stop_words = stopwords.words('english') #What are the stopwords to use?
book_sentiment, cleaned_chaps = book_sentiment_wrapper(book_dir, book_id, stop_words)

chap_list = []
for i in range(len(cleaned_chaps)):
    chap_elem = ' '.join(cleaned_chaps[i])
    chap_list.append(chap_elem)

#Make pandas dataframe. Each row is a chapter
df_pos = pd.DataFrame(chap_list, columns=['chapter_contents'])

#Add column with Chapter names. 
chapter_names = ['Preface']
i = 0
for i in range(len(cleaned_chaps)- 1):
    chap_to_add = str(i+1)
    chapter_names.append(chap_to_add)
    
#Make series for book id since I'll be processing multiple books later   
book_id_col = []
i = 0
for i in range(len(cleaned_chaps)):
    book_id_col.append(book_id.replace('\\', '').replace('.txt', ''))

df_pos.insert(0, column='Book_ID', value = book_id_col)    
df_pos.insert(1, column='Chapter_Name', value = chapter_names)

chap_word_count = []
for i in range(df_pos.shape[0]):
    chapter_name = df_pos['Chapter_Name'][i]
    word_count   = len(df_pos['chapter_contents'][i].split())
    chap_word_count.append(word_count)

max_words = max(chap_word_count)
max_index = chap_word_count.index(max_words)
max_chap  = df_pos['Chapter_Name'][max_index]

sid = SentimentIntensityAnalyzer() # register vader sid data
df_pos['compound'] = [sid.polarity_scores(x)['compound'] for x in df_pos['chapter_contents']]
df_pos['neg']      = [sid.polarity_scores(x)['neg'] for x in      df_pos['chapter_contents']]
df_pos['neu']      = [sid.polarity_scores(x)['neu'] for x in      df_pos['chapter_contents']]
df_pos['pos']      = [sid.polarity_scores(x)['pos'] for x in      df_pos['chapter_contents']]

text_object = NRCLex(df_pos['chapter_contents'][0]) # register nrc lexicon data
scores = text_object.raw_emotion_scores
df_pos['joy']          = [NRCLex(x).raw_emotion_scores['joy']          for x in df_pos['chapter_contents']]
df_pos['positive']     = [NRCLex(x).raw_emotion_scores['positive']     for x in df_pos['chapter_contents']]
df_pos['anticipation'] = [NRCLex(x).raw_emotion_scores['anticipation'] for x in df_pos['chapter_contents']]
df_pos['sadness']      = [NRCLex(x).raw_emotion_scores['sadness']      for x in df_pos['chapter_contents']]
df_pos['surprise']     = [NRCLex(x).raw_emotion_scores['surprise']     for x in df_pos['chapter_contents']]
df_pos['negative']     = [NRCLex(x).raw_emotion_scores['negative']     for x in df_pos['chapter_contents']]
df_pos['anger']        = [NRCLex(x).raw_emotion_scores['anger']        for x in df_pos['chapter_contents']]
df_pos['disgust']      = [NRCLex(x).raw_emotion_scores['disgust']      for x in df_pos['chapter_contents']]
df_pos['trust']        = [NRCLex(x).raw_emotion_scores['trust']        for x in df_pos['chapter_contents']]
df_pos['fear']         = [NRCLex(x).raw_emotion_scores['fear']         for x in df_pos['chapter_contents']]

df_pos.insert(loc=len(df_pos.columns), column='Chapter_WC', value = chap_word_count)
df_pos['joy_n']          = df_pos['joy']/df_pos['Chapter_WC']           # normalize by word count
df_pos['positive_n']     = df_pos['positive']/df_pos['Chapter_WC']
df_pos['anticipation_n'] = df_pos['anticipation']/df_pos['Chapter_WC']
df_pos['sadness_n']      = df_pos['sadness']/df_pos['Chapter_WC']
df_pos['surprise_n']     = df_pos['surprise']/df_pos['Chapter_WC']
df_pos['negative_n']     = df_pos['negative']/df_pos['Chapter_WC']
df_pos['anger_n']        = df_pos['anger']/df_pos['Chapter_WC']
df_pos['disgust_n']      = df_pos['disgust']/df_pos['Chapter_WC']
df_pos['trust_n']        = df_pos['trust']/df_pos['Chapter_WC']
df_pos['fear_n']         = df_pos['fear']/df_pos['Chapter_WC']