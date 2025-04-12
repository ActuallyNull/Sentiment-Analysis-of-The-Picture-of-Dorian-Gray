import matplotlib.pyplot as plt
from lemmatizing_denoising import denoised_chapter
from nltk import FreqDist, classify, NaiveBayesClassifier

freq_dist_pos = FreqDist(denoised_chapter)
print('These are the 20 most common words in Chapter 1:')
print(freq_dist_pos.most_common(20))
fig = plt.figure(figsize=(12,8))
freq_dist_pos.plot(20, cumulative=False,color = 'purple', linestyle = ':', marker='.', markersize=16)
plt.title('Word Frequency Distribution of Chapter 1')
plt.show()