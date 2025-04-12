import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
from sentimentAnalysis import *

mask = np.array(Image.open("book.jpg"))

# generate the word cloud for the positive tweets   
WC = WordCloud(
                          max_words=5000,
                          mask = mask,
                          contour_width=2,
                          max_font_size=150,
                          font_step=2,
                          background_color='black',
                          width=298,
                          height=169
                          ).generate(str(chap_list))

fig = plt.figure(figsize=(25,30))
plt.axis("off")
plt.imshow(WC)
plt.title('Word Cloud')
plt.tight_layout(pad=0)
plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()