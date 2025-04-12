from sentimentAnalysis import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n','fear_n']
df_pos_nrc = df_pos[col_names]

sns.set_context("paper", rc={"axes.labelsize":24})
sns.pairplot(df_pos_nrc, height=2.5)
plt.suptitle("Pair Plots of Every Sentiment", fontsize=40, y=1.02)
plt.savefig("plots/pair_plot_sentiments.png", format='png', bbox_inches='tight')