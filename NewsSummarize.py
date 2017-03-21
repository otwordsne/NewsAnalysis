"""
This will go to the links of the page relevant to your specified homepage. now bloomberg
Give you the top 5 nouns, verbs, adjectives, or top 15 words.
Give you the articles with sentiment above a certain threshold magnitude
"""
from bs4 import BeautifulSoup
#import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
#from gensim.summarization import summarize
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

def get_soup(url):
    #opener = AppURLopener()
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    html = urllib.request.urlopen(req)
    #response = opener.open(url)
    soup = BeautifulSoup(html)
    return soup

# ul class = "hfwmm-list hfwmm-4uphp-list-hfwmm-light-list", data-track-previs = "flex4uphphero"
def get_usatoday_headlines(url="http://www.usatoday.com"):
    # html = urllib.urlopen(url)
    soup = get_soup(url)
    headlines = soup.find('ul', attrs = {"class":"hfwmm-list hfwmm-4uphp-list hfwmm-light-list"})

    if headlines is None:
        print('No headlines found.')
        return

    extensions = headlines.findAll("a")
    top_article_links = [story['href'] for story in extensions if "watch-live" not in story['href']]
    headline_urls = [url + exten for exten in top_article_links if not exten.startswith("http")]
    headline_soups = [get_soup(link) for link in headline_urls]
    headline_texts = [" ".join([ptag.get_text() for ptag in s.findAll("p")]) for s in headline_soups]
    cleaned_texts = [strip_usatoday(article) for article in headline_texts]
    return cleaned_texts

extensions = headlines.findAll("a")
top_article_links = [story['href'] for story in extensions if "watch-live" not in story['href']]
headline_urls = [url + exten for exten in top_article_links if not exten.startswith("http")]
headline_soups = [get_soup(link) for link in headline_urls]
headline_texts = [" ".join([ptag.get_text() for ptag in s.findAll("p")]) for s in headline_soups]
cleaned_texts = [re.sub("^[\w|\s]+\n","",article) for article in headline_texts]
def strip_usatoday(text):
    garbage = ["Settings Cancel Set Hi  Already a subscriber? Subscribe to USA TODAY Already a print edition subscriber, but don't have a login? Manage your account settings. Log Out  Get the news Let friends in your social network know what you are reading about", "A link has been posted to your Facebook feed.", "To find out more about Facebook commenting please read the Conversation Guidelines and FAQs", "A link has been sent to your friend\'s email address"]
    for s in garbage:
        text = text.replace(s, "")
#    text = text.replace(garbage, "")
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
    headline = re.sub('^.*\/','', re.sub('\/$','',url)).replace("'",'').replace('-',' ')
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
    text = [re.sub("('|\n)", '', x) for x in text]  # |(\n)+.*(\n)+)
    text = ' '.join(p for p in text)
    text = str(text.encode('ascii', 'ignore')).replace('-','')  # ignore the unencodable strings
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


#def get_keep_words():
#    keepwords = set(['s&p500', '&'])
#    return keepwords


# Removes the stopwords from
def remove_stopwords(text):
    text = keep_words(text)
    stop = stopwords
    text = text.lower()
    # tokenizer = get_tokenizer("en_US")
    tokenizer = RegexpTokenizer(r'\w+\&*\w*')
    cleaned_tokens = [str(word) for word in tokenizer.tokenize(text) if word not in stop]
    # cleaned_tokens = [word for word in tokens if word not in stopwords]
    return cleaned_tokens
# Split text by sentence
def splitBySentence(text):
    endPunc = re.compile('[.!]')
    sentList = endPunc.split(text)
    return sentList

# Count the number of words occurrences in the tokenized text
def countWords(tokenizedText):
    wordCounts = {}
    for word in tokenizedText:
        if word in wordCounts.keys():
            wordCounts[word] += 1
        else:
            wordCounts[word] = 1
    wordCounts = sorted(wordCounts.items(), key=operator.itemgetter(1))
    return wordCounts

# Score a sentence by a list of top words
def scoreSentByTopWords(sentence, topWords):
    score = 0
    for word in topWords:
        if word in sentence:
            score += sentence.count(word)
    return score

# Score a sentence by a list of top words according to how heavily the word is weighted
def scoreSentByWordWeights(sentence, topWordsDict):
    score = 0
    for word in topWordsDict.keys():
        if word in sentence:
            score += (sentence.count(word) * topWordsDict[word])
    return score

# Find the sentence with the minimum score in a diction of sentences with scores
def findMinSent(sentScoreDict):
   min = ('', -1)
   for sent in sentScoreDict.keys():
       if (sentScoreDict[sent] < min[1] or min[1] == -1):
           minSent = (sent, sentScoreDict[sent])
   return minSent

# Summary text by the score of top words (dict of top words with weight) in each sentence
def summarizeByTopWords(text, topWords, sentsInSummary = 3):
    sentScores = {}
    sentences = splitBySentence(text)
    topSents = {}
    #max = 0
    for sent in sentences:
        #score = scoreSentByTopWords(sent, topWords)
        score = scoreSentByWordWeights(sent, topWords)
        if sent in sentScores.keys():
            sentScores[sent] += score
        else:
            sentScores[sent] = score
 
        if (len(topSents.keys()) < sentsInSummary):
            topSents[sent] = sentScores[sent]
        else:
            minSent = findMinSent(topSents)
            if (sentScores[sent] >= minSent[1]):
                topSents[sent] = sentScores[sent]
                del topSents[minSent[0]]
    return ' '.join(list(topSents.keys()))
            
# wordCountDict is sorted ascending order
def summary(text, wordCountDict, numWordsToUse = 5, sentsInSummary = 3):
    #if numWordsToUse < 1:
    assert(numWordsToUse > 1), "Number of words to score with must be greater than 1"
    topWords = wordCountDict[-1*(numWordsToUse):-1]
    topWords.append(wordCountDict[-1])
    topWords = dict(topWords)
    return summarizeByTopWords(text, topWords, sentsInSummary)
    


def summarize(text, topWordsTuples, numWordsToScore = 5):
    return summarizeByTopWords(text,[word[0] for word in topWordsTuples[0:numWordsToScore]]) 
        

def clean_text(text):
    #text = re.sub('\u2014','--', text)
    return remove_stopwords(text)
    # text = remove_links(text)


def main():
    url1 = 'https://www.bloomberg.com/news/articles/2017-01-06/wall-street-is-starting-to-get-nervous-about-all-the-money-pouring-into-u-s-stocks'
    url2 = 'https://www.bloomberg.com/news/articles/2017-01-05/japan-shares-to-drop-on-yen-gain-u-s-jobs-loom-markets-wrap'
    article1 = get_bloomberg_text(url1)
    article2 = get_bloomberg_text(url2)

    tokenized_txt1 = remove_stopwords(article1[1])
    tokenized_txt2 = remove_stopwords(article2[1])

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
