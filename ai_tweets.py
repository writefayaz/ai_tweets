# Author Fayaz Mammoo

import pandas as pd
import numpy as np
from time import sleep
import tweepy
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import re, string, unicodedata 
from PIL import Image
import matplotlib.pyplot as plt 

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []                        # Create empty list to store pre-processed words.
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)        # Append processed words to new list.
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []                        # Create empty list to store pre-processed words.
    for word in words:
        new_word = word.lower()           # Converting to lowercase
        new_words.append(new_word)        # Append processed words to new list.
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []                        # Create empty list to store pre-processed words.
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)    # Append processed words to new list.
    return new_words

def remove_stopwords(words,stopwords):
    """Remove stop words from list of tokenized words"""
    new_words = []                        # Create empty list to store pre-processed words.
    for word in words:
        if word not in stopwords:
            new_words.append(word)        # Append processed words to new list.
    return new_words

def lemmatize_list(words,lemmatizer):
    new_words = []
    for word in words:
      new_words.append(lemmatizer.lemmatize(word))
    return new_words

def pre_process_tweets(t,combined_re,regex_pattern,token,stopwords,lemmatizer):
    del_amp = BeautifulSoup(t, 'lxml')
    del_amp_text = del_amp.get_text()
    del_link_mentions = re.sub(combined_re, '', del_amp_text)
    del_emoticons = re.sub(regex_pattern, '', del_link_mentions)
    lower_case = del_emoticons.lower()
    words = token.tokenize(lower_case)
    result_words = [x for x in words if len(x) > 2]
    return (" ".join(words)).strip()

def tweets_to_csv(keyword,recent,api): 
    try:
        tweets = tweepy.Cursor(api.search,
                           q=keyword,
                           count=100,
                           result_type="recent",
                           include_entities=True,
                           lang="en").items(recent)
        tweets_list = [[tweet.text] for tweet in tweets]
        df = pd.DataFrame(tweets_list,columns=['Text'])
        df.to_csv('{}.csv'.format(keyword), sep=',', index = False)
    except BaseException as e:
        print('failed on_status,',str(e))
        sleep(3)

def main():
    consumer_key        = ""
    consumer_secret     = ""
    access_token        = ""
    access_token_secret = ""

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

    keyword = '#programming'+ "-filter:retweets"
    recent  = 3000
    tweets_to_csv(keyword, recent,api)

    df = pd.read_csv("./#programming-filter:retweets.csv")
    pd.options.display.max_colwidth = 200
    df.head()
    
    re_list = ['(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?', '@[A-Za-z0-9_]+','#']
    combined_re = re.compile( '|'.join( re_list) )

    regex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              "]+", flags = re.UNICODE)

    token = WordPunctTokenizer()
    stopwords = set(STOPWORDS)
    stopwords.update(["programming","coding","software","development","developer"])


    print("Pre-Processing ...\n")
    cleaned_tweets = []
    for i in range(0,recent):
        if( (i+1)%100 == 0 ):
            print("Tweets {} of {} have been processed".format(i+1,3000))                                                                  
        cleaned_tweets.append(pre_process_tweets((df.Text[i]),combined_re,regex_pattern,token,stopwords,lemmatizer))

    string_words = pd.Series(cleaned_tweets).str.cat(sep=' ')
    
    wordcloud = WordCloud(width=1600, stopwords=stopwords,height=800,max_font_size=200,max_words=50,collocations=False, background_color='grey').generate(string_words)
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    #Saving to File
    wordcloud.to_file("wc.png")

    # Uploading PNG Image File
    #tweetImg = api.media_upload("wc.png")
    # Posting Wordcloud Image
    #tweet = "This is an AI bot tweeting Word Cloud from recent programming trends!"
    #post_ = api.update_status(status=tweet, media_ids=[tweetImg.media_id])


if __name__ == '__main__':
    main()