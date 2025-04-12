#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis on the Literary Works of Oscar Wilde
# Oscar Wilde is one of my favourite authors ever. 'The Picture of Dorian Gray ' published in 1890, remains as one of the stories that has resonated and impacted me the most. The idea of a painting granting you a carte blanche and being the receptacle of ones evil while you live a life devoid of consequences was fascinating to me. In particular, the gradual deterioration of the painting as Dorian Gray committed increasingly terrible acts will always be one of my favorite concepts in literature. 
# 
# <figure class="half" style="display:flex">
#     <img src="https://upload.wikimedia.org/wikipedia/commons/9/92/Oscar_Wilde_3g07095u-adjust.jpg" width = 200 height = 400>
#     <img src="http://hilsingermendelson.com/wp-content/uploads/2018/08/cover.jpg" width = 200 height = 400>
#     <img src="https://sites.google.com/site/thepictureofdoriangray2/_/rsrc/1357874556041/historic-context/dorian.jpg" width = 475 height = 400>
# </figure>
# 
# As such, it's only natural for me to want to investigate them through the lenses of data science and natural language processing (NLP) techniques. 
# 
# NLP is an amalgam of the concepts and techniques from the fields of linguistics, computer science, and artificial intelligence that seeks to understand how computers process human language. Some common tasks used by NLP are text and speech processing used in Speech recognition applications (i.e., Siri from Apple and Alexa from Amazon) and Higher-level applications like Discourse management (i.e., how can a computer converse with a human) amongst many others. It is an absolutely fascinating field and one that I've been deeply interested in from the moment I learned about it.
# 
# Some questions I'd want to answer are: 
# 
# * What are the most commonly used words used by Oscar Wilde?
# * What is the overall sentiment present in his works?
# * How does the sentiment change (if at all) from chapter to chapter? Is there a particular emotional structure that is present?
# * Are the emotional assessments in agreement with the themes present in the narratives?
# 
# I'll write some code to download his works from <a href = "https://www.gutenberg.org/">Project Gutenberg</a>. These routines should be readily applicable to get stuff from other authors too!
# 
# The current workflow/structure of my process is as follows:
# 
# 1. Programatically download books from Project Gutenberg. 
# 2. Programatically split the books into Chapters/Sections that can be subsequently processed. 
# 3. Tokenize, lemmatize, and denoise the book contents to prepare them for NLP analysis. 
# 4. Implement a Naive Bayes Classifier, the VADER and NRC lexicons to get progressively more detailed emotional breakdowns of the sentiments present in each chapter (joy, positive, anger, disgust, negative, anticipation, surprise, trust, fear, and surprise). 
# 5. Analyze the resulting polarity/emotion scores for each chapter/section on the book and see how they interplay with one another over the course of the story. 
# 
# My ultimate goal is to start generating a bit of an 'emotional database' for a variety of writers that could help me understand the way different authors express the quantified sentiments in their work. Then, with this, maybe one could start developing an algorithm/AI that could take the emotional/thematic elements from each writer and try to replicate their writing style. It would be pretty neat for instance to eventually generate a database of the sentiments of a variety of gothic horror writers like Mary Shelley, Bram Stoker, and Edgar Allan Poe and see how their stories are constructed. What are the common themes? What are the common sentiments? And can we parse/recognize these things through code?
# 
# With that said let's get started!
# 
# ## Modules used in this work
# The following modules were used to get this project done

# In[21]:

from webscraping import * #Import all methods from methods.py
#Modules for downloading stuff off Project Gutenberg
import os, requests
from os.path import basename
from os.path import join
from os import makedirs
from urllib.request import urlopen
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from bs4 import BeautifulSoup
import re

#Stuff for general sentiment analysis
import nltk
from nltk.corpus import twitter_samples,stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

#More stuff for sentiment analysis
nltk.download('punkt') #Pretrained model to tokenize words
nltk.download('wordnet') #Lexical database to help determine base word
nltk.download('averaged_perceptron_tagger') #Used to determine context of word in sentence
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

#Stuff for VADER
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#For NRC lexicon
from nrclex import NRCLex

#Stuff for dealing with strings and regular expressions
import re, string, random

#General modules
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

#Machine Learning Library for Python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#Performance metrics for generated model
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.metrics import roc_curve,classification_report
from scikitplot.metrics import plot_confusion_matrix

#For WordCloud visualizations
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image

# ## Wrapper function for Performing Sentiment Analysis on Book
# Let me compile my routines and subsequently process all the chapters in the book and see what's the overall sentiment in each chapter and the book using this model.


# However, I want to see if a more 'colorful' chapter assessment can be made using Emotion Detection/Recognition through VADER and NRC lexicons.
# 
# ## Sentiment Analysis using VADER
# VADER will give polarity scores. Even though the sentiment categorization is still set to a simple positive, negative, or neutral, we will get a quantifiable metric for each of those values and start getting a better idea of the sentiment composition from each chapter. I'll first modify my wrapper routine from above so that it returns the chapter contents as part of the output.



# Now I'll rerun my function on the entire book to get the chapter contents. These chapter contents have already been tokenized, normalized and denoised.



# And now I'll run the SentimentIntensityAnalyzer using VADER and see what we get

# In[39]:
df_pos = ()

sid = SentimentIntensityAnalyzer()
df_pos['compound'] = [sid.polarity_scores(x)['compound'] for x in df_pos['chapter_contents']]
df_pos['neg']      = [sid.polarity_scores(x)['neg'] for x in      df_pos['chapter_contents']]
df_pos['neu']      = [sid.polarity_scores(x)['neu'] for x in      df_pos['chapter_contents']]
df_pos['pos']      = [sid.polarity_scores(x)['pos'] for x in      df_pos['chapter_contents']]
df_pos


# Okay, now things are starting to get a bit more interesting. Each chapter has different positive, negative and neutral scores associated with them. Furthermore, the compound scores (which represent the overall sentiment with +1 being the most positive and -1 being the most negative) are also different between chapters. Let me plot them and see how it looks like!

# In[ ]:


fig, axs = plt.subplots(2,2,figsize=(30,25))
# scatter plot
nRows = 2
nCols = 2
col_names = ['neg', 'neu', 'pos', 'compound']
axs_names = ['negative', 'neutral', 'positive', 'compound']
i = 0
j = 0
counter = 0
for val in range(len(col_names)):
    col_name = col_names[val]
    axs_name = axs_names[val]
    r = lambda: random.randint(0,255)
    hexcol = '#%02X%02X%02X' % (r(),r(),r())
    axs[i,j].plot(df_pos['Chapter_Name'], 
                  df_pos[col_name],
                  marker='o',  
                  markersize=12, 
                  linewidth=4,
                  color= hexcol)
    axs[i,j].set_xticks(df_pos['Chapter_Name'])
    axs[i,j].set_yticks([df_pos[col_name].min(), df_pos[col_name].max()])
    axs[i,j].set_xticklabels(df_pos['Chapter_Name'], rotation=90 )
    axs[i,j].set_yticklabels([df_pos[col_name].min(), df_pos[col_name].max()])
    axs[i,j].set_ylabel(axs_name, fontsize = 30)
    axs[i,j].grid(True)
    if i < nRows-1:
        i += 1
    else:
        i = 0
        j += 1

plt.show()


# ## NRC Lexicon Categorization
# Now let's try using the NRC lexicon which has 10 different emotions it can categorize text data into. The emotions are: joy, positive, anticipation, sadness, surprise, negative, andger, disgust, trust, and fear. The NRC lexicon is composed of over 27,000 words at the time of writing. You can find more info on the NRC lexicon here: <a href = "http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm">http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm</a>
# 
# To use it you need to first install it via <code>pip install NRCLex</code>
# 
# Then you can import it into your environment via <code>from nrclex import NRCLex</code>

# In[ ]:


text_object = NRCLex(df_pos['chapter_contents'][0])
scores = text_object.raw_emotion_scores
scores


# Let me start by calculating the raw emotion scores for each chapter and adding them into the dataframe. The result of this will give me the number of words in the chapter that are associated with one of the 10 emotions listed above.

# In[ ]:


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
df_pos


# Nice! Now we have a lot more data to work with! Let's plot each of the sentiment frequencies for each of the chapters.

# In[ ]:


fig, axs = plt.subplots(10,1,figsize=(30,40))
# scatter plot
col_names = ['joy', 'positive', 'anticipation', 'sadness',
             'surprise', 'negative', 'anger', 'disgust', 'trust','fear']
i = 0
counter = 0

cmap = plt.cm.get_cmap('twilight_shifted')
slicedCM = cmap(np.linspace(0, 1, len(col_names))) 

for val in range(len(col_names)):
    col_name = col_names[val]
    #Make a random hex color
    r = lambda: random.randint(0,255)
    hexcol = '#%02X%02X%02X' % (r(),r(),r())
    axs[i].plot(df_pos['Chapter_Name'], 
                df_pos[col_name], 
                marker='o',  
                markersize=12, 
                linewidth=4,  
                color= slicedCM[val])
    axs[i].set_xticks(df_pos['Chapter_Name'])
    axs[i].set_yticks([df_pos[col_name].min(), df_pos[col_name].max()],fontsize = 20)
    axs[i].set_xticklabels(df_pos['Chapter_Name'], rotation=90 , fontsize = 16)
    axs[i].set_yticklabels([df_pos[col_name].min(), df_pos[col_name].max()] , fontsize = 30)
    axs[i].set_ylabel(col_name, fontsize = 30)
    axs[i].grid(True)
    i+=1
plt.show()


# Hmm, interesting... the overall line shape for the sentiments looks pretty close to one another throughout the book. There is a spike in emotion happening for Chapter 11 however. Why? I wonder if it's as simple of an explanation as Chapter 11 being significantly longer than all the other chapters. Let's see...

# In[ ]:


chap_word_count = []
for i in range(df_pos.shape[0]):
    chapter_name = df_pos['Chapter_Name'][i]
    word_count   = len(df_pos['chapter_contents'][i].split())
    chap_word_count.append(word_count)
    print(chapter_name, ' has ', word_count, ' words')

max_words = max(chap_word_count)
max_index = chap_word_count.index(max_words)
max_chap  = df_pos['Chapter_Name'][max_index]
print('The chapter with the most words has ',max_words, ' words')
print('The chapter with the most words is at index ', max_index)
print('The chapter with the most words is ', max_chap)


# Indeed. Chapter 11 is longer than all the other chapters and as such its word count is inflating the results. I will repeat the analysis but I will normalize the results by the chapter word count and see if that changes things.
# I'll start by adding a chapter word count column at the end of the dataframe

# In[ ]:


df_pos.insert(loc=len(df_pos.columns), column='Chapter_WC', value = chap_word_count)
df_pos.head()


# And now, I'll generate new columns with the chapter word_count normalized values. 

# In[ ]:


df_pos['joy_n']          = df_pos['joy']/df_pos['Chapter_WC']
df_pos['positive_n']     = df_pos['positive']/df_pos['Chapter_WC']
df_pos['anticipation_n'] = df_pos['anticipation']/df_pos['Chapter_WC']
df_pos['sadness_n']      = df_pos['sadness']/df_pos['Chapter_WC']
df_pos['surprise_n']     = df_pos['surprise']/df_pos['Chapter_WC']
df_pos['negative_n']     = df_pos['negative']/df_pos['Chapter_WC']
df_pos['anger_n']        = df_pos['anger']/df_pos['Chapter_WC']
df_pos['disgust_n']      = df_pos['disgust']/df_pos['Chapter_WC']
df_pos['trust_n']        = df_pos['trust']/df_pos['Chapter_WC']
df_pos['fear_n']         = df_pos['fear']/df_pos['Chapter_WC']
df_pos.head()


# Let me plot these results now and see what's up

# In[ ]:


import numpy as np
fig, axs = plt.subplots(10,1,figsize=(30,40))
# scatter plot
col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']
i = 0
counter = 0


cmap = plt.cm.get_cmap('twilight_shifted')
slicedCM = cmap(np.linspace(0, 1, len(col_names))) 

for val in range(len(col_names)):
    col_name = col_names[val]
    #Make a random hex color
    #r = lambda: random.randint(0,255)
    #hexcol = '#%02X%02X%02X' % (r(),r(),r())
    axs[i].plot(df_pos['Chapter_Name'], 
                df_pos[col_name], 
                marker='o',  
                markersize=12, 
                linewidth=4,  
                color= slicedCM[val])

    axs[i].set_yticks([df_pos[col_name].min(), df_pos[col_name].max()],fontsize = 20)
    axs[i].set_yticklabels(np.round([df_pos[col_name].min(), df_pos[col_name].max()],2) , fontsize = 30)
    axs[i].set_ylabel(col_name, fontsize = 30)
    axs[i].grid(True)
    i+=1
axs[i-1].set_xticks(df_pos['Chapter_Name'])
axs[i-1].set_xticklabels(df_pos['Chapter_Name'], rotation=90 , fontsize = 30)
plt.show()


# That's pretty interesting! The different emotions have different behaviors over the course of the book! Are they somehow related? Let me generate a Pairplot from these results before discussing them further. I'll generate a new dataframe from the main dataframe that will contain the results from the NRC lexicon emotion analysis.

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']
df_pos_nrc = df_pos[col_names]
#df_pos_nrc.insert(0, column='Chapter_Name', value = df_pos['Chapter_Name'])
df_pos_nrc.head()


# And now, I'll generate the Pairplot

# In[ ]:


sns.set_context("paper", rc={"axes.labelsize":24})
sns.pairplot(df_pos_nrc, height=2.5)


# Neat! Some of the emotions seem to be related to one another! For instance, the joy and positive scores are correlated to one another which makes sense. The disgust and fear scores on the other hand seem to be anti-correlated to the joy score. Let me generate the correlation matrix for these emotions to be sure.
# 

# In[ ]:


f = plt.figure(figsize=(19, 15))
fignum=f.number
plt.matshow(df_pos_nrc.corr(), fignum=f.number)
plt.xticks(range(df_pos_nrc.select_dtypes(['number']).shape[1]),
           df_pos_nrc.select_dtypes(['number']).columns,
           fontsize=20,
           rotation=45)
plt.yticks(range(df_pos_nrc.select_dtypes(['number']).shape[1]),
           df_pos_nrc.select_dtypes(['number']).columns, 
           fontsize=20)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.title('Correlation Matrix for Emotions in The Portrait of Dorian Gray', fontsize=40)
plt.show()


# The correlation matrix shows how correlated the emotion scores are to one another. The values range from +1 (signifying high <strong>positive</strong> correlation) and -1 (signifying high <strong>negative</strong> correlation). When the color is brighter (more yellow), the two emotions being compared are more closely associated to one another. When the color is darker (more purple), the two emotions being compared are more distant to one another. 
# 
# From this matrix we can see which emotions are most/least closely related to one another in 'The Portrait of Dorian Gray'. For instance, joy seems to be very closely related to anticipation while fear is the most distant emotion to joy.  Disgust and Fear seem to be the antithesis of positive emotions in this book which I would argue are thematically accurate given the repulsion and fear that Dorian Gray experiences over the course of the book due to the nature of his persona and the events that unfolded which are made physically manifest through the painting that Basil made of him.
# ## Discussion of Results
# Let's start looking at the behavior of each of the emotions measured over the course of the book.
# 
# I'll be displaying results using the 'plasma' color palette shown below where the change in color signifies book progression. For each emotion I'll also be plotting the relationship between it and the two emotions most similar/dissimilar to it.
# 
# ### Joy
# Joy is defined as 'a feeling of great pleasure and happiness'. Joy appears to have a fairly consistent downward trend over the course of the book. It has an approximately positive linear relationship with the positive sentiment and an approximately negative linear relationship with fear.

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize = (20,8))
sns.barplot(data = df_pos,
             x = df_pos['Chapter_Name'],
             y = df_pos['joy_n'],
             palette = 'plasma')
plt.xticks(rotation=90 , fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Joy Score',fontsize = 30)
plt.xlabel('')
plt.show()

from scipy import stats
#Regression plot 1
r, p = stats.pearsonr(df_pos['joy_n'], df_pos['positive_n'])
a0 = sns.jointplot(x="joy_n", y="positive_n", data=df_pos, kind='reg', fit_reg=True, ci = 90)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 1
sns.jointplot(x="joy_n", y="positive_n", data=df_pos, kind='kde',shade=True)

#Regression plot 3
r, p = stats.pearsonr(df_pos['joy_n'], df_pos['fear_n'])
a2 = sns.jointplot(x="joy_n", y="fear_n"    , data=df_pos, kind='reg')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 3
sns.jointplot(x="joy_n", y="fear_n"    , data=df_pos, kind='kde',shade=True)
plt.show()


# ### Positive
# Positive sentiments are defined as 'a good, affirmative, or constructive quality or attribute'. Positive sentiments also appear to have a downward trend for the first 13 chapters of the book. After that however, the positive sentiments start to trend upwards but they never reach the initial levels. 

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize = (20,8))
sns.barplot(data = df_pos,
             x = df_pos['Chapter_Name'],
             y = df_pos['positive_n'],
             palette = 'plasma')
plt.xticks(rotation=90 , fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Positive Score',fontsize = 30)
plt.xlabel('')
plt.show()

from scipy import stats
#Regression plot 1
r, p = stats.pearsonr(df_pos['positive_n'], df_pos['anticipation_n'])
a0 = sns.jointplot(x="positive_n", y="anticipation_n", data=df_pos, kind='reg', fit_reg=True, ci = 90)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 1
sns.jointplot(x="positive_n", y="anticipation_n", data=df_pos, kind='kde',shade=True)

#Regression plot 3
r, p = stats.pearsonr(df_pos['positive_n'], df_pos['disgust_n'])
a2 = sns.jointplot(x="positive_n", y="disgust_n"    , data=df_pos, kind='reg')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 3
sns.jointplot(x="positive_n", y="disgust_n"    , data=df_pos, kind='kde',shade=True)
plt.show()


# ### Anticipation
# Anticipation is defined as 'excitement about something that's going to happen'. Anticipation appears to behave in an oscillatory fashion. However, each 'anticipation peak' is less strong than the last.

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize = (20,8))
sns.barplot(data = df_pos,
             x = df_pos['Chapter_Name'],
             y = df_pos['anticipation_n'],
             palette = 'plasma')
plt.xticks(rotation=90 , fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Anticipation Score',fontsize = 30)
plt.xlabel('')
plt.show()

from scipy import stats
#Regression plot 1
r, p = stats.pearsonr(df_pos['anticipation_n'], df_pos['joy_n'])
a0 = sns.jointplot(x="anticipation_n", y="joy_n", data=df_pos, kind='reg', fit_reg=True, ci = 90)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 1
sns.jointplot(x="anticipation_n", y="disgust_n", data=df_pos, kind='kde',shade=True)

#Regression plot 3
r, p = stats.pearsonr(df_pos['anticipation_n'], df_pos['disgust_n'])
a2 = sns.jointplot(x="anticipation_n", y="disgust_n"    , data=df_pos, kind='reg')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 3
sns.jointplot(x="anticipation_n", y="disgust_n"    , data=df_pos, kind='kde',shade=True)
plt.show()


# ### Sadness
# Sadness is defined as 'an emotional pain associated with, or characterized by, feelings of disadvantage, loss, despair, grief, helplessness, disappointment and sorrow'. Sadness remains fairly consistent over the course of the book. It does seem to trend downwards at the end.

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize = (20,8))
sns.barplot(data = df_pos,
             x = df_pos['Chapter_Name'],
             y = df_pos['sadness_n'],
             palette = 'plasma')
plt.xticks(rotation=90 , fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Sadness Score',fontsize = 30)
plt.xlabel('')
plt.show()

from scipy import stats
#Regression plot 1
r, p = stats.pearsonr(df_pos['sadness_n'], df_pos['anticipation_n'])
a0 = sns.jointplot(x="sadness_n", y="anticipation_n", data=df_pos, kind='reg', fit_reg=True, ci = 90)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 1
sns.jointplot(x="sadness_n", y="anticipation_n", data=df_pos, kind='kde',shade=True)

#Regression plot 3
r, p = stats.pearsonr(df_pos['sadness_n'], df_pos['surprise_n'])
a2 = sns.jointplot(x="sadness_n", y="surprise_n"    , data=df_pos, kind='reg')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 3
sns.jointplot(x="sadness_n", y="surprise_n"    , data=df_pos, kind='kde',shade=True)
plt.show()


# ### Surprise
# Surprise is defined as 'to cause to feel wonder or amazement because of being unexpected'. Surprise also behaves in a very similar way to sadness within the context of this book.

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize = (20,8))
sns.barplot(data = df_pos,
             x = df_pos['Chapter_Name'],
             y = df_pos['surprise_n'],
             palette = 'plasma')
plt.xticks(rotation=90 , fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Surprise Score',fontsize = 30)
plt.xlabel('')
plt.show()

from scipy import stats
#Regression plot 1
r, p = stats.pearsonr(df_pos['surprise_n'], df_pos['positive_n'])
a0 = sns.jointplot(x="surprise_n", y="positive_n", data=df_pos, kind='reg', fit_reg=True, ci = 90)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 1
sns.jointplot(x="surprise_n", y="positive_n", data=df_pos, kind='kde',shade=True)

#Regression plot 3
r, p = stats.pearsonr(df_pos['surprise_n'], df_pos['anticipation_n'])
a2 = sns.jointplot(x="surprise_n", y="anticipation_n"    , data=df_pos, kind='reg')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 3
sns.jointplot(x="surprise_n", y="anticipation_n"    , data=df_pos, kind='kde',shade=True)
plt.show()


# ### Negative
# Negative is defined as 'marked by absence, withholding, or removal of something positive '. Negative sentiments start relatively low and remains steady until it peaks in Chapters 16 and 18 before decaying again

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize = (20,8))
sns.barplot(data = df_pos,
             x = df_pos['Chapter_Name'],
             y = df_pos['negative_n'],
             palette = 'plasma')
plt.xticks(rotation=90 , fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Negative Score',fontsize = 30)
plt.xlabel('')
plt.show()

from scipy import stats
#Regression plot 1
r, p = stats.pearsonr(df_pos['negative_n'], df_pos['anger_n'])
a0 = sns.jointplot(x="negative_n", y="anger_n", data=df_pos, kind='reg', fit_reg=True, ci = 90)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 1
sns.jointplot(x="negative_n", y="anger_n", data=df_pos, kind='kde',shade=True)

#Regression plot 3
r, p = stats.pearsonr(df_pos['negative_n'], df_pos['fear_n'])
a2 = sns.jointplot(x="negative_n", y="fear_n"    , data=df_pos, kind='reg')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 3
sns.jointplot(x="negative_n", y="fear_n"    , data=df_pos, kind='kde',shade=True)
plt.show()


# ### Anger
# Anger is defined as 'a strong feeling of displeasure or annoyance and often of active opposition to an insult, injury, or injustice'. Anger behaves in the same way as Negative sentiments. It starts relatively low and remains steady until it peaks in Chapters 16 and 18 before decaying again.

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize = (20,8))
sns.barplot(data = df_pos,
             x = df_pos['Chapter_Name'],
             y = df_pos['anger_n'],
             palette = 'plasma')
plt.xticks(rotation=90 , fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Anger Score',fontsize = 30)
plt.xlabel('')
plt.show()

from scipy import stats
#Regression plot 1
r, p = stats.pearsonr(df_pos['anger_n'], df_pos['negative_n'])
a0 = sns.jointplot(x="anger_n", y="negative_n", data=df_pos, kind='reg', fit_reg=True, ci = 90)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 1
sns.jointplot(x="anger_n", y="negative_n", data=df_pos, kind='kde',shade=True)

#Regression plot 3
r, p = stats.pearsonr(df_pos['anger_n'], df_pos['disgust_n'])
a2 = sns.jointplot(x="anger_n", y="disgust_n"    , data=df_pos, kind='reg')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 3
sns.jointplot(x="anger_n", y="disgust_n"    , data=df_pos, kind='kde',shade=True)
plt.show()


# ### Disgust
# Disgust is defined as 'a feeling of revulsion or strong disapproval aroused by something unpleasant or offensive'. Disgust seems to be gradually trending upwards until it peaks in CHapter 16. Then, it starts to decay until the end of the book.

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize = (20,8))
sns.barplot(data = df_pos,
             x = df_pos['Chapter_Name'],
             y = df_pos['disgust_n'],
             palette = 'plasma')
plt.xticks(rotation=90 , fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Disgust Score',fontsize = 30)
plt.xlabel('')
plt.show()

from scipy import stats
#Regression plot 1
r, p = stats.pearsonr(df_pos['disgust_n'], df_pos['anger_n'])
a0 = sns.jointplot(x="disgust_n", y="anger_n", data=df_pos, kind='reg', fit_reg=True, ci = 90)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 1
sns.jointplot(x="disgust_n", y="anger_n", data=df_pos, kind='kde',shade=True)

#Regression plot 3
r, p = stats.pearsonr(df_pos['disgust_n'], df_pos['fear_n'])
a2 = sns.jointplot(x="disgust_n", y="fear_n"    , data=df_pos, kind='reg')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 3
sns.jointplot(x="disgust_n", y="fear_n"    , data=df_pos, kind='kde',shade=True)
plt.show()


# ### Trust
# Trust is defined as the 'firm belief in the reliability, truth, ability, or strength of someone or something'. Trust trends upwards until Chapter 7. After Chapter 7, it drops and remains low for the rest of the book.

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize = (20,8))
sns.barplot(data = df_pos,
             x = df_pos['Chapter_Name'],
             y = df_pos['trust_n'],
             palette = 'plasma')
plt.xticks(rotation=90 , fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Trust Score',fontsize = 30)
plt.xlabel('')
plt.show()

from scipy import stats
#Regression plot 1
r, p = stats.pearsonr(df_pos['trust_n'], df_pos['joy_n'])
a0 = sns.jointplot(x="trust_n", y="joy_n", data=df_pos, kind='reg', fit_reg=True, ci = 90)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 1
sns.jointplot(x="trust_n", y="joy_n", data=df_pos, kind='kde',shade=True)

#Regression plot 3
r, p = stats.pearsonr(df_pos['trust_n'], df_pos['fear_n'])
a2 = sns.jointplot(x="trust_n", y="fear_n"    , data=df_pos, kind='reg')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 3
sns.jointplot(x="trust_n", y="fear_n"    , data=df_pos, kind='kde',shade=True)
plt.show()


# ### Fear
# Fear is defined as 'an unpleasant emotion caused by the belief that someone or something is dangerous, likely to cause pain, or a threat'. Fear creeps upwards throughout most of the book. Peaking in Chapters 16 and 18 after which it starts to decay until the end.

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize = (20,8))
sns.barplot(data = df_pos,
             x = df_pos['Chapter_Name'],
             y = df_pos['fear_n'],
             palette = 'plasma')
plt.xticks(rotation=90 , fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Fear Score',fontsize = 30)
plt.xlabel('')
plt.show()

from scipy import stats
#Regression plot 1
r, p = stats.pearsonr(df_pos['fear_n'], df_pos['disgust_n'])
a0 = sns.jointplot(x="fear_n", y="disgust_n", data=df_pos, kind='reg', fit_reg=True, ci = 90)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 1
sns.jointplot(x="fear_n", y="disgust_n", data=df_pos, kind='kde',shade=True)

#Regression plot 3
r, p = stats.pearsonr(df_pos['fear_n'], df_pos['positive_n'])
a2 = sns.jointplot(x="fear_n", y="positive_n"    , data=df_pos, kind='reg')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

#KDE plot 3
sns.jointplot(x="fear_n", y="positive_n"    , data=df_pos, kind='kde',shade=True)
plt.show()


# Let me plot them together now.

# In[ ]:


col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']

fig = plt.figure(figsize=(30, 20))
colors = plt.cm.rainbow(np.linspace(0, 1, 30))
plt.plot(df_pos["Chapter_Name"], 
         df_pos[col_names],
         marker='o',  
         markersize=12, 
         linewidth=4,
         label = col_names)
plt.xticks(fontsize = 30,rotation = 90)
plt.yticks(fontsize = 30)
plt.ylabel('Sentiments',fontsize = 30)
plt.legend(fontsize = 25)
plt.grid(True)
plt.show()


# Interesting. The positive sentiment remains dominant throughout most of the book. It is overtaken by the negative sentiment in Chapters 10, 13, 14, 16 and 18 though. Sadness is on par with the positive sentiment on Chapter 13 as well. The book appears to end on a positive note.
# 
# ## Word Cloud for The Portrait of Dorian Gray
# Here's a neat visualization for the top words in The Portrait of Dorian Gray. The bigger the word size the more frequently it appears and viceversa. It appears some of the most common words in the book are life, know, one, go and look.

# In[ ]:





# ## Conclusions
# Sentiment Analysis and Emotion Recognition was used to analyze 'The Portrait of Dorian Gray' by Oscar Wilde. The results of the routine showcased here elucidated some of the thematic/emotional structure of that book using a Naive Bayes classifier, VADER and NRC lexicons for every chapter in the book. The next steps for this project involve processing the remaining works from Oscar Wilde and seeing if there is a common emotional structure in his books.
