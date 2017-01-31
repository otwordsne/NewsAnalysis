"""
This will go to the links of the page relevant to your specified homepage. now bloomberg
Give you the top 5 nouns, verbs, adjectives, or top 15 words.
Give you the articles with sentiment above a certain threshold magnitude
"""
from bs4 import BeautifulSoup
#import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from gensim.summarization import summarize
import re
import requests
import urllib.request
from urllib.request import Request, urlopen

stopwords = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by',
                'for', 'from', 'has', 'he,', 'in', 'is', 'it',
                'its', "it's", 'of', 'on', 'that', 'the', 'to',
                'was', 'were', 'will', 'with', 'you', 'have', 'been',
                'her', 'hers', 'she', 'him', 'his', 'my', 'mine',
                'this', 'these', 'their', 'theirs', 'them', 'we'])

class URL:
    def __init__( self, url):
        self.url = url

    def __str__( self):
        return self.url

    def get_soup( self, url):
        html = urllib.request.urlopen(url)
        soup = BeautifulSoup(html)
        return soup

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

urllib._urlopener = AppURLopener()

def get_soup(url):
    #opener = AppURLopener()
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    html = urllib.request.urlopen(req)
    #response = opener.open(url)
    soup = BeautifulSoup(html)
    return soup

def get_usnews_headlines(url='http://www.usnews.com/'):
    # html = urllib.urlopen(url)
    soup = get_soup(url)
    headlines = soup.findall('div', attrs={'class': 'flex-media block-normal small-middle \
                                       medium-top, border-bottom-for-small-only \
                                       padding-bottom-normal-for-small-only'})
    if headlines is None:
        print('Article not found.')
        return

    text = [x.get_text() for x in headlines]
    text = [re.sub('\n', '', x) for x in text]
    text = [re.sub('\.', '', x) for x in text]
    # text = ' '.join(p for p in text)
    return text


def get_googlenews_headlines(url='http://www.new.google.com/'):
    # html = urllib.urlopen(url)
    soup = get_soup(url)
    headlines = soup.findall('h2', attrs={'class': 'esc-lead-article-title'})
    if headlines is None:
        print('Article not found.')
        return

    text = [x.get_text() for x in headlines]
    text = [re.sub('\n', '', x) for x in text]
    text = [re.sub('\.', '', x) for x in text]
    # text = ' '.join(p for p in text)
    return text


def get_bloomberg_text(url):
    # html = urllib.urlopen(url)
    headline = re.sub('^.*\/','', re.sub('\/$','',url)).replace("'",'').replace(".","").replace('-',' ')
    soup = get_soup(url)
    body = soup.find('div', attrs={'class': 'body-copy'})
    if body is None:
        body = soup.find_all('div', attrs={'class': 'main__section'})
        text_w_tags = [x.find_all('p', recursive=False) for x in body]
        text_w_tags = [x for sublist in text_w_tags for x in sublist]

        if body is None:
            print('Article not found.')
            return
    else:
        text_w_tags = body.find_all('p', recursive=False)
    text = [x.get_text() for x in text_w_tags if not x.find(class_='inline-newsletter')]
    if text == '':
        return
    # text = [x.get_text() for x in text]
    text = [re.sub("('|\n|\.)", '', x) for x in text]  # |(\n)+.*(\n)+)
    text = ' '.join(p for p in text)
    text = str(text.encode('ascii', 'replace')).replace('?',' ').replace('-','')
    return (headline, text)  # text  # (headline, text)

def get_bloomberg_articles(url='https://bloomberg.com/'):
    soup = get_soup(url)
    links = soup.find_all('a')
    extensions = set(['www.bloomberg.com/news',
                      'www.bloomberg.com/politics',
                      'www.bloomberg.com/features'])
    articles = set()
    for x in links:
        try:
            if any(valid_article in x['href'] for valid_article in extensions):
                if 'video' not in x['href']:
                    articles.add(x['href'])
        except Exception:
            pass
    # articles = [x['href'] for x in links if ('www.bloomberg/news' or 'www.bloomberg/politics') in x['href']]
    return articles

def get_bloomberg_page_articles_text(url='https://www.bloomberg.com/'):
    article_urls = get_bloomberg_articles(url)
    texts = [get_bloomberg_text(x) for x in article_urls]
    texts = [x for x in texts if x is not None]
    return texts

def get_all_bloomberg(main_extensions = ['politics', 'markets', 'technology']):
    home = 'https://www.bloomberg.com/'
    main_pages = [home]
    for name in main_extensions:
        main_pages.append(home + name)

    # Find breaking news first
    # breaking = find('h1', attrs= {'class': 'breaking-news-banner__headline'})
    # breaking_headline = breaking.get_text()
    page_dict = {}
    page_dict['homepage'] = get_bloomberg_page_articles_text(home)
    for url in main_extensions:
        page_dict[url] = get_bloomberg_page_articles_text(home + url)

    # texts = [(url,get_bloomberg_page_articles_text(url)) for url in mainpages]
    # texts = [x for sublist in texts for x in sublist]
    return page_dict

def keep_words(text):
    text = text.lower()
    text = re.sub('s&p 500', 's&p500', text)
    return text


def get_keep_words():
    keepwords = set(['s&p500', '&'])
    return keepwords


# Removes the stopwords from
def remove_stopwords(text):
    text = keep_words(text)
    keepwords = get_keep_words()
    # stop = set(stopwords.words('english'))
    # stop = stop.difference(keepwords)
    stop = stopwords
    text = text.lower()
    # tokenizer = get_tokenizer("en_US")
    tokenizer = RegexpTokenizer(r'\w+\&*\w*')
    cleaned_tokens = [str(word) for word in tokenizer.tokenize(text) if word not in stop]
    # cleaned_tokens = [word for word in tokens if word not in stopwords]
    return cleaned_tokens


'''
def remove_links(text):
    webpage = re.compile(r'\s\S*\.[gov|com|org]')
    weird = re.compile('\s\S*\:\S*\s?')
    text = re.sub(webpage, '', text)
    text = re.sub(weird, '', text)
    return text
    # tokens = [toke for toke in tokens if not (webpage.match(toke) or weird.match(toke))]
    # return tokens
'''

def clean_text(text):
    text = re.sub('\u2014','--', text)
    text = remove_stopwords(text)
    # text = remove_links(text)

from nltk.corpus import wordnet

def get_wordnet_pos(word_tag_tuple):
    tag = word_tag_tuple[1]
    if tag.startswith('J'):
        tag = wordnet.ADJ
    elif tag.startswith('V'):
        tag = wordnet.VERB
    elif tag.startswith('N'):
        tag = wordnet.NOUN
    elif tag.startswith('R'):
        tag = wordnet.ADV
    else:
        tag = ''
    return (word_tag_tuple[0], tag)

def main():
    url1 = 'https://www.bloomberg.com/news/articles/2017-01-06/wall-street-is-starting-to-get-nervous-about-all-the-money-pouring-into-u-s-stocks'
    url2 = 'https://www.bloomberg.com/news/articles/2017-01-05/japan-shares-to-drop-on-yen-gain-u-s-jobs-loom-markets-wrap'
    article1 = get_bloomberg_text(url1)
    article2 = get_bloomberg_text(url2)

    tokenized_txt1 = remove_stopwords(article1)
    tokenized_txt2 = remove_stopwords(article2)

    tagged_txt1 = nltk.pos_tag(tokenized_txt1)
    tagged_txt2 = nltk.pos_tag(tokenized_txt2)
    # print "hi"
    # body = [x.find('div', attrs={'class': 'body-copy'}) for x in article]
    # text_w_tags = body.find_all('p')
    # text = [x.get_text() for x in text_w_tags]
    print(article1)
    print(tokenized_txt1)

    # docs = [tokenized_txt2]
    # dictionary = corpora.Dictionary(docs)
    # doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
    # ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=10, id2word = dictionary)
    # print(ldamodel.print_topics(num_topics=3, num_words=3))

if __name__ == main():
    main()
