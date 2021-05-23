%matplotlib inline

import datetime
import json

import numpy as np

# For plotting
import matplotlib.pyplot as plt

# NLTK's sentiment code
import nltk
import nltk.sentiment.util
import nltk.sentiment.vader

# TextBlob provides its own sentiment analysis
from textblob import TextBlob

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
    currentTime = currentTime.replace(second=0, minute = 0)

    # If our frequency map already has this time, use it, otherwise add
    extended_list = rel_frequency_map.get(currentTime, [])
    extended_list.append(tweet)
    rel_frequency_map[currentTime] = extended_list
    
# Fill in any gaps
times = sorted(rel_frequency_map.keys())
print (len(times))
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

# Sentiment values
polarVals_tb = []
objVals = []

# For each minute, pull the tweet text and search for the keywords we want
for t in sortedTimes:
    tweets = rel_frequency_map[t]
    
    # For calculating averages
    localPolarVals = []
    localObjVals = []
    
    for tweet in tweets:
        tweetString = tweet["text"]

        blob = TextBlob(tweetString)
        polarity = blob.sentiment.polarity
        objectivity = blob.sentiment.subjectivity
        
        localPolarVals.append(polarity)
        localObjVals.append(objectivity)
        
    # Add data to the polarity and objectivity measure arrays
    if ( len(tweets) > 0 ):
        polarVals_tb.append(np.mean(localPolarVals))
        objVals.append(np.mean(localObjVals))
    else:
        polarVals_tb.append(0.0)
        objVals.append(0.0)

        
# Now plot this sentiment data
fig, ax = plt.subplots()
fig.set_size_inches(11, 8.5)

plt.title("Sentiment")
plt.xticks(smallerXTicks, [sortedTimes[x] for x in smallerXTicks], rotation=90)

xData = range(len(sortedTimes))

# Polarity is scaled [-1, 1], for negative and positive polarity
ax.plot(xData, polarVals_tb, label="Polarity")

# Subjetivity is scaled [0, 1], with 0 = objective, 1 = subjective
ax.plot(xData, objVals, label="Subjectivity")

ax.legend()
ax.grid(b=True, which=u'major')

plt.show()

vader = nltk.sentiment.vader.SentimentIntensityAnalyzer()

# Sentiment values
polarVals_vader = []

# For each minute, pull the tweet text and search for the keywords we want
for t in sortedTimes:
    tweets = rel_frequency_map[t]
    
    # For calculating averages
    localPolarVals = []
    
    for tweet in tweets:
        tweetString = tweet["text"]

        polarity = vader.polarity_scores(tweetString)["compound"]
        
        localPolarVals.append(polarity)
        
    # Add data to the polarity and objectivity measure arrays
    if ( len(tweets) > 0 ):
        polarVals_vader.append(np.mean(localPolarVals))
    else:
        polarVals_vader.append(0.0)

        
# Now plot this sentiment data
fig, ax = plt.subplots()
fig.set_size_inches(11, 8.5)

plt.title("Sentiment")
plt.xticks(smallerXTicks, [sortedTimes[x] for x in smallerXTicks], rotation=90)

xData = range(len(sortedTimes))

# Polarity is scaled [-1, 1], for negative and positive polarity
ax.plot(xData, polarVals_vader, label="Polarity")

ax.legend()
ax.grid(b=True, which=u'major')

plt.ylim((-0.95, 0.95))
plt.show()

# Now plot this sentiment data
fig, ax = plt.subplots()
fig.set_size_inches(11, 8.5)

plt.title("Sentiment")
plt.xticks(smallerXTicks, [sortedTimes[x] for x in smallerXTicks], rotation=90)

xData = range(len(sortedTimes))

# Polarity is scaled [-1, 1], for negative and positive polarity
ax.plot(xData, polarVals_vader, label="VADER")
ax.plot(xData, polarVals_tb, label="TextBlob")

ax.legend()
ax.grid(b=True, which=u'major')

plt.ylim((-0.95, 0.95))
plt.show()

top_k = 10

# Calculate sentiment for each relevant tweet
sentiment_pairs = [(tweet, vader.polarity_scores(tweet["text"])["compound"]) 
                   for tweet in relevant_tweets]

sorted_tweets = sorted(sentiment_pairs, key=lambda x: x[1])

# Most negative tweets
for tweet, sentiment in sorted_tweets[:top_k]:
    print("Author:", tweet["user"]["screen_name"], "Sentiment:", sentiment)
    print("Text:\n%s" % tweet["text"], "\n")
    
    # Most negative tweets
for tweet, sentiment in sorted_tweets[-top_k:]:
    print("Author:", tweet["user"]["screen_name"], "Sentiment:", sentiment)
    print("Text:\n%s" % tweet["text"], "\n")
    
    
# Calculate sentiment for each relevant tweet
sentiment_pairs = [(tweet, TextBlob(tweet["text"]).sentiment.polarity) 
                   for tweet in relevant_tweets]
​
sorted_tweets = sorted(sentiment_pairs, key=lambda x: x[1])
​
print("Most Negative Tweets:")
# Most negative tweets
for tweet, sentiment in sorted_tweets[:top_k]:
    print("Author:", tweet["user"]["screen_name"], "Sentiment:", sentiment)
    print("Text:\n%s" % tweet["text"], "\n")
    
print("------------------------")
print("Most Positive Tweets:")
# Most positive tweets
for tweet, sentiment in sorted_tweets[-top_k:]:
    print("Author:", tweet["user"]["screen_name"], "Sentiment:", sentiment)
    print("Text:\n%s" % tweet["text"], "\n")
    
    