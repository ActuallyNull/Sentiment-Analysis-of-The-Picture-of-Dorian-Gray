from nltk.tokenize import word_tokenize
from chapterCategorization import get_book_chapters, chapter_list
from nltk.tag import pos_tag

chapter = get_book_chapters(chapter_list, 1) #Get the contents of chapter 2

def tokenize_chapter_contents(contents):
    chap_token = word_tokenize(contents)
    del(chap_token[0:2])

    return chap_token

chap_token = tokenize_chapter_contents(chapter) # tokenising the text into words, punctuation will be removed during the denoising process
#print(pos_tag(chap_token)[0:20]) # tagging the tokens with their part of speech, meaning: https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
