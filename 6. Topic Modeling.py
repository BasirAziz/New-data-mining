%matplotlib inline

import datetime
import json
import string
import os

import numpy as np

# For plotting
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

crisisInfo = {
    "Women's March": {
        "name": "Women's March 2017",
        "time": 1484992800, # 21 January 2017, 6:58 UTC to 08:11 UTC
        "directory": "womensmarch",    # Where do we find the relevant files
        "keywords": [    # How can we describe this event?
            "women's march","resist", "notmypresident","inauguration","women's right","human right","planned parenthood"
        ],
        "place": [
            38.899539,# Latitude
            -77.036551 # Longitude
        ],
        "box": {    # Where did this event occur?
            "lowerLeftLon": -77.119759,
            "lowerLeftLat": 38.791645,
            "upperRightLon": -76.909393,
            "upperRightLat": 38.995548,
        }
    },
}

# Replace the name below with your selected crisis
selectedCrisis = "Women's March"




in_file_path = "/Users/yutingliao/Desktop/INST728 E/relevant_tweet_output_keywords_updated.json" # Replace this as necessary

relevant_tweets = []
with open(in_file_path, "r") as in_file:
    for line in in_file:
        relevant_tweets.append(json.loads(line.encode("utf8")))
        
print("Relevant Tweets:", len(relevant_tweets))

# Twitter's time format, for parsing the created_at date
timeFormat = "%a %b %d %H:%M:%S +0000 %Y"

# Frequency map for tweet-times
rel_frequency_map = {}
for tweet in relevant_tweets:
    # Parse time
    currentTime = datetime.datetime.strptime(tweet['created_at'], timeFormat)

    # Flatten this tweet's time
    currentTime = currentTime.replace(second=0)

    # If our frequency map already has this time, use it, otherwise add
    extended_list = rel_frequency_map.get(currentTime, [])
    extended_list.append(tweet)
    rel_frequency_map[currentTime] = extended_list
    
# Fill in any gaps
times = sorted(rel_frequency_map.keys())
firstTime = times[0]
lastTime = times[-1]
thisTime = firstTime

# We want to look at per-minute data, so we fill in any missing minutes
timeIntervalStep = datetime.timedelta(0, 60)    # Time step in seconds
while ( thisTime <= lastTime ):

    rel_frequency_map[thisTime] = rel_frequency_map.get(thisTime, [])
        
    thisTime = thisTime + timeIntervalStep

# Count the number of minutes
print ("Start Time:", firstTime)
print ("Stop Time:", lastTime)
print ("Processed Times:", len(rel_frequency_map))

fig, ax = plt.subplots()
fig.set_size_inches(11, 8.5)

plt.title("Tweet Frequencies")

sortedTimes = sorted(rel_frequency_map.keys())
postFreqList = [len(rel_frequency_map[x]) for x in sortedTimes]

smallerXTicks = range(0, len(sortedTimes), 90)
plt.xticks(smallerXTicks, [sortedTimes[x] for x in smallerXTicks], rotation=90)

xData = range(len(sortedTimes))

ax.plot(xData, postFreqList, color="blue", label="Posts")

ax.grid(b=True, which=u'major')
ax.legend()

plt.show()

# This list comprehension iterates through the tweet_list list, and for each
#  tweet, it iterates through the hashtags list
htags = [
        hashtag["text"].lower() 
         for tweet in relevant_tweets 
             for hashtag in tweet["entities"]["hashtags"]
        ]

print("\nTotal Hashtag Count:", len(htags))
print("Unique Hashtag Count:", len(set(htags)))

htags_freq = nltk.FreqDist(htags)

print("\nFrequent Hashtags:")
for tag, count in htags_freq.most_common(20):
    print(tag, count)
    
    # Gotta pull in a bunch of packages for this

# Actual LDA implementation
import gensim.models.ldamulticore

# Actual ATM implementation
import gensim.models.atmodel

# CountVectorizer turns tokens into numbers for us
from sklearn.feature_extraction.text import CountVectorizer

# Gensim models
from gensim.corpora import Dictionary  # All the words that appear in our dataset
from gensim.models import TfidfModel # For down-weighting frequent tokens
from gensim.models.phrases import Phrases # For building bigrams

# But first, read in stopwrods
enStop = stopwords.words('english')
frStop = stopwords.words('french')
esStop = stopwords.words('spanish')

# Skip stop words, retweet signs, @ symbols, and URL headers
stopList = enStop +\
    frStop + esStop +\
    ["http", "https", "rt", "@", ":", "co", "amp", "&amp;", "...", "\n", "\r"] +\
    crisisInfo[selectedCrisis]["keywords"]
stopList.extend(string.punctuation)

vectorizer = CountVectorizer(strip_accents='unicode', 
                             tokenizer=TweetTokenizer(preserve_case=False).tokenize,
                             stop_words=stopList)

# Build the Analyzer
analyze = vectorizer.build_analyzer() 

# For each tweet, tokenize it according to the CountVectorizer
analyzed_text = [analyze(tweet["text"]) for tweet in relevant_tweets]

# As an example, note the removed stopwords
print(relevant_tweets[0]["text"])
print(analyzed_text[0])


# Make bigrams from the text, but only for really common bigrams
bigram = Phrases(analyzed_text, min_count=5)
bi_analyzed_text = [bigram[x] for x in analyzed_text]

# As an example, note the removed stopwords
print(relevant_tweets[0]["text"])
print(analyzed_text[0])
print(bi_analyzed_text[0])


# Build a dictionary from this text
dictionary = Dictionary(bi_analyzed_text)

# Filter out words that occur too frequently or too rarely.
# Disregarding stop words, this dataset has a very high number of low frequency words.
max_freq = 0.75
min_count = 5
dictionary.filter_extremes(no_below=min_count, no_above=max_freq)

# This sort of "initializes" dictionary.id2token.
_ = dictionary[0]

# Create a map for vectorizer IDs to words
id2WordDict = dictionary.id2token
word2IdDict = dict(map(lambda x: (x[1], x[0]), id2WordDict.items()))

# Create a bag of words
corpus = [dictionary.doc2bow(text) for text in analyzed_text]

# Train TFIDF model
tfidf_model = TfidfModel(corpus)

# Built TFIDF-transformed corpus
tfidf_corpus = [tfidf_model[text] for text in corpus]


k = 10

lda = gensim.models.LdaMulticore(tfidf_corpus, 
                                 id2word=id2WordDict,
                                 num_topics=k) # ++ iterations for better results

ldaTopics = lda.show_topics(num_topics=k, 
                            num_words=10, 
                            formatted=False)

for (i, tokenList) in ldaTopics:
    print ("Topic %d:" % i, ' '.join([pair[0] for pair in tokenList]))
    print()
    
    
    import pyLDAvis.gensim

pyLDAvis.enable_notebook()


pyLDAvis.gensim.prepare(lda, tfidf_corpus, dictionary)


# Simple pipepline for analyzing tweet text
def analysis_pipeline(text):
    a1 = analyze(text)
    a2 = bigram[a1]
    a3 = dictionary.doc2bow(a2)
    a4 = tfidf_model[a3]

    return a4

analyzed_tweet_pairs = list(
    filter(lambda x: len(x[0]) > 0,
           [(analysis_pipeline(tweet["text"]), tweet["user"]["id"]) 
            for tweet in relevant_tweets])
)

atm_docs = [x[0] for x in analyzed_tweet_pairs]
doc_to_author = dict([(x, [y[1]]) for x, y in enumerate(analyzed_tweet_pairs)])


k = 10

atm = gensim.models.atmodel.AuthorTopicModel(corpus=atm_docs, 
                                             id2word=id2WordDict,
                                             doc2author=doc_to_author,
                                             num_topics=k) # ++ iterations for better results

atmTopics = atm.show_topics(num_topics=k, 
                            num_words=10, 
                            formatted=False)

for (i, tokenList) in atmTopics:
    print ("Topic %d:" % i, ' '.join([pair[0] for pair in tokenList]))
    print()
    
    
    topic_counter = {x:[0]*len(rel_frequency_map) for x in range(lda.num_topics)}
    
    
    for (i, d) in enumerate(rel_frequency_map.keys()):
    tweets = rel_frequency_map[d]
    
    for tweet in tweets:
        text = tweet["text"]
        topic_dist = lda.get_document_topics(analysis_pipeline(text))
        
        top_topic = sorted(topic_dist, key=lambda x: x[1])[-1][0]
        
        topic_counter[top_topic][i] += 1
        
        
        fig, ax = plt.subplots()
fig.set_size_inches(11, 8.5)

plt.title("Tweet Frequencies")

smallerXTicks = range(0, len(sortedTimes), 90)
plt.xticks(smallerXTicks, [sortedTimes[x] for x in smallerXTicks], rotation=90)

xData = range(len(sortedTimes))

for this_k in range(lda.num_topics):
    plt.plot(xData, topic_counter[this_k], label="Topic %d" % (this_k))

ax.grid(b=True, which=u'major')
ax.legend()

plt.show()


