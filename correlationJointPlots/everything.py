import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from scipy import stats
import numpy as np

# Access sentimentAnalysis.py one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentimentAnalysis import df_pos


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
plt.tight_layout()
plt.title('All Sentiment Analysis of Chapters', fontsize=40)
plt.savefig("plots/correlationJointPlots/everything.png", format='png', bbox_inches='tight')
plt.show()