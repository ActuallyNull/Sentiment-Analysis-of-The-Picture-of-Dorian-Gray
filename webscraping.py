from urllib.request import urlopen
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import re
import os, requests
from os.path import basename
from os.path import join
from os import makedirs
# In[2]:


# download a file from a URL, returns content of downloaded file
def download_url(urlpath):
    try:
        # open a connection to the server
        with urlopen(urlpath, timeout=3) as connection:
            # read the contents of the url as bytes and return it
            return connection.read()
    except:
        return None


# In[3]:
def getFirstLastName(author_name):
    aname = author_name.lower().rsplit()
    first_name = aname[0]
    last_name = aname[1]

    return first_name, last_name

author_name = 'Oscar Wilde'
firstname, lastname = getFirstLastName(author_name)
print(firstname, lastname)

# In[4]:


# decode downloaded html and extract all <a href=""> links
def get_urls_from_html(content):

    # decode the provided content as ascii text
    html = content.decode('utf-8')

    # parse the document as best we can
    soup = BeautifulSoup(html, 'html.parser')

    # find all all of the <a href=""> tags in the document
    atags = soup.find_all('a')

    #Get links from search query page
    #nlinks = len(atags) - 25 - 5 
    #del(atags[0:5]) #Remove first six links since they don't correspond to books 
    #del(atags[-nlinks:])#Remove links that aren't book results (this query displays 25 books)

    # get all links from a tags
    return [tag.get('href') for tag in atags]



# In[5]:


# return all book unique identifiers from a list of raw links
def get_book_identifiers(links):
    # define a url pattern we are looking for
    pattern = re.compile('/ebooks/[0-9]+')
    # process the list of links for those that match the pattern
    books = set()
    for link in links:
        # check of the link matches the pattern
        if not pattern.match(link):
            continue
        # extract the book id from /ebooks/nnn
        book_id = link[8:]
        # store in the set, only keep unique ids
        books.add(book_id)
    return books


# In[6]:

# download one book from project gutenberg
def download_book(book_id, save_path):
    print(save_path)
    # construct the download url
    url = f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
    # download the content
    data = download_url(url)
    if data is None:
        #print(f'Failed to download {url}')
        url = f'https://www.gutenberg.org/files/{book_id}/{book_id}.txt'
        data = download_url(url)
        if data is None:
            return f'Failed to download {url}'
        else:
            # create local path
            save_file = join(save_path, f'{book_id}.txt')
            # save book to file
            with open(save_file, 'wb') as file:
                file.write(data)
            return f'Saved {save_file}'

    # create local path
    save_file = join(save_path, f'{book_id}.txt')
    # save book to file
    with open(save_file, 'wb') as file:
        file.write(data)
    return f'Saved {save_file}'

# In[8]:


def download_all_books(url, save_path):
    # download the page that lists top books
    data = download_url(url)
    print(f'.downloaded {url}')
    # extract all links from the page
    links = get_urls_from_html(data)
    print(f'.found {len(links)} links on the page')
    # retrieve all unique book ids
    book_ids = get_book_identifiers(links)
    print(f'.found {len(book_ids)} unique book ids')
    # create the save directory if needed
    makedirs(save_path, exist_ok=True)
    # download and save each book in turn
    for book_id in book_ids:
        print(book_id)
        # download and save this book
        result = download_book(book_id, save_path)
        # report result
        print(result)

# In[7]:

#Get current working directory and download book into 'books' directory
cwd = os.getcwd()
book_dir = cwd + '\\' + 'books'
#test = download_book('174', book_dir) download the picture of dorian gray w/ id 174
author_name = 'oscar wilde'
first_name, last_name = getFirstLastName(author_name)
author_url = f'https://www.gutenberg.org/ebooks/search/?query={first_name}+{last_name}&submit_search=Search'

download_all_books(author_url, book_dir)