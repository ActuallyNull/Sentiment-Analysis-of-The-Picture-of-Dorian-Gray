import re
import os
import pandas as pd

book_id = '\\174.txt'           #What book are we processing?
book_dir = 'books'              #Directory where the book is located
book_name = book_dir + book_id  #Location of book

# opens text file as 'book' in read mode
def split_book_into_chapters(book_dir, book_id):

    book_name = book_dir + book_id #Location of book

    #Open the book
    book = open(book_name, "r", encoding="utf8") 
    #Assign the book a name as string
    book = str(book.read()) 
    #Use regex to split the book into chapters by finding instances of the word CHAPTER
    chapters = re.split("CHAPTER ", book)
    #Remove first 21 CHAPTER instances since they are just fluff
    del(chapters[0:20])
    #Loops for the number of chapters in the book, starting at chapter 1
    for i in range(1, len(chapters)+1):
        book_chapter = open("chapters/{}.txt".format(i), "w+", encoding="utf8") #Make a new book
        book_chapter.write(chapters[i-1]) #Write on book with current chapter content
        book_chapter.close() #Closes the book

    chapter_list = []
    for file in os.listdir("chapters"):
        if file.endswith(".txt"):
            chapter_list.append(file)

    #Do a natural sort on the texts files 
    chapter_list.sort(key=lambda x: '{0:0>8}'.format(x).lower())
    return chapter_list

#print(chapter[0:248])
def get_book_chapters(chapter_list, chapter_num):

    chapter_to_read = chapter_list[chapter_num]

    #Open text file containing book chapter and read contents
    with open("chapters/"+chapter_to_read, encoding = 'utf-8') as f:
        chapter_contents = f.read().rstrip()

    return chapter_contents

chapter_list = split_book_into_chapters(book_dir, book_id) #Get the list of chapters from the book

num_chapters = len(chapter_list)
chapter_names = []
df_list = []
i = 0
for file in chapter_list:
    chapter_name = 'Chapter ' + str(i)
    chapter_names.append(chapter_name)
    df = pd.read_csv('chapters/'+file,
                     sep = 'delimiter', 
                     header = 0,
                     names = [chapter_name],
                     encoding='utf-8',
                     on_bad_lines='skip',
                     engine='python'
                    )
    df_list.append(df)
    i += 1

df_tot = pd.concat(df_list, axis =1, names = chapter_names)
#print(df_tot.head(3))