%matplotlib inline

import datetime
import json
import gzip
import glob
import os
import string

import numpy as np

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

crisisInfo = {
    "Women's March": {
        "name": "Women's March 2017",
        "time": 1484992800, # 21 January 2017, 6:58 UTC to 08:11 UTC
        "directory": "womensmarch",    # Where do we find the relevant files
        "keywords": [    # How can we describe this event?
            "women's march","resist", "notmypresident","inauguration","women's right","human right","planned parenthood"
        ],
        "box": {    # Where did this event occur?
            "lowerLeftLon": 2.54563,
            "lowerLeftLat": 49.496899,
            "upperRightLon": 6.40791,
            "upperRightLat": 51.5050810,
        }
    },
}

selectedCrisis = "Women's March"

# Determine host-specific location of data
tweetDirectory = crisisInfo[selectedCrisis]["directory"]
tweetGlobPath = os.path.sep + os.path.join("Users", "yutingliao", "Desktop", "INST728 E",
                             "Dataset_women'smarch",tweetDirectory,"part-00*.gz")

print ("Reading files from:", tweetGlobPath)

# Dictionary for mapping dates to data
frequencyMap = {}

# Twitter's time format, for parsing the created_at date
timeFormat = "%a %b %d %H:%M:%S +0000 %Y"

for tweetFilePath in glob.glob(tweetGlobPath):
    print ("Reading File:", tweetFilePath)

    for line in gzip.open(tweetFilePath, 'rb'):

        # Convert from bytes to UTF8 string
        decoded_line = line.decode("utf8")
        
        # Try to read tweet JSON into object
        tweetObj = None
        try:
            tweetObj = json.loads(decoded_line)
        except json.JSONDecodeError as jde:
            print("JSON Decode Error:", decoded_line)
            continue

        # Deleted status messages and protected status must be skipped
        if ( "delete" in tweetObj.keys() or "status_withheld" in tweetObj.keys() ):
            continue

        # Try to extract the time of the tweet
        try:
            currentTime = datetime.datetime.strptime(tweetObj['created_at'], timeFormat)
        except:
            print("Error parsing time on line:", decoded_line)
            raise

        # Flatten this tweet's time
        currentTime = currentTime.replace(second=0)

        # If our frequency map already has this time, use it, otherwise add
        extended_list = frequencyMap.get(currentTime, [])
        extended_list.append(tweetObj)
        frequencyMap[currentTime] = extended_list
        
        # Fill in any gaps
times = sorted(frequencyMap.keys())
firstTime = times[0]
lastTime = times[-1]
thisTime = firstTime

# We want to look at per-minute data, so we fill in any missing minutes
timeIntervalStep = datetime.timedelta(0, 60)    # Time step in seconds
while ( thisTime <= lastTime ):

    frequencyMap[thisTime] = frequencyMap.get(thisTime, [])
        
    thisTime = thisTime + timeIntervalStep

# Count the number of minutes
print ("Start Time:", firstTime)
print ("Stop Time:", lastTime)
print ("Processed Times:", len(frequencyMap))
    
# Count all the tweets per minute
print ("Processed Tweet Count:", np.sum([len(x) for x in frequencyMap.values()]))

# Get data about our crisis
selectedCrisis = "Women's March"

crisisMoment = crisisInfo[selectedCrisis]["time"] # When did it occur by epoch time
crisisTime = datetime.datetime.utcfromtimestamp(crisisMoment) # Convert to datetime
crisisTime = crisisTime.replace(second=0) # Flatten to a specific minute

# Print converted time
print ("Event Time:", crisisTime)

# Create a new figure in which to plot minute-by-minute tweet count
fig, ax = plt.subplots()
fig.set_size_inches(11, 8.5)

plt.title("Tweet Frequency")

# Sort the times into an array for future use
sortedTimes = sorted(frequencyMap.keys())

# What time span do these tweets cover?
print ("Time Frame:", sortedTimes[0], sortedTimes[-1])

# Get a count of tweets per minute
postFreqList = [len(frequencyMap[x]) for x in sortedTimes]

# We'll have ticks every few minutes (more clutters the graph)
smallerXTicks = range(0, len(sortedTimes), 30)
plt.xticks(smallerXTicks, [sortedTimes[x] for x in smallerXTicks], rotation=90)

# Plot the post frequency
ax.plot(range(len(frequencyMap)), postFreqList, color="blue", label="Posts")

# Get the index for when our crisis occurred
crisisXCoord = sortedTimes.index(crisisTime)
ax.scatter([crisisXCoord], [np.mean(postFreqList)], c="r", marker="x", s=200, label="Event")

ax.grid(b=True, which=u'major')
ax.legend()

ax.set_xlabel("Post Date")
ax.set_ylabel("Twitter Frequency")

plt.show()

# List of relevant tweet times
after_event_times = []

relevant_time_span = 60 # Let's look at the first few minutes after the event

current_time_span = 0
for current_time in sortedTimes:
    if ( current_time >= crisisTime ):
        after_event_times.append(current_time)
        current_time_span += 1
        
    if ( current_time_span > relevant_time_span ):
        break
    
    # Create a new figure in which to plot minute-by-minute tweet count
fig, ax = plt.subplots()
fig.set_size_inches(11, 8.5)

plt.title("Temporally Relevant Tweet Frequency")

# Get a count of tweets per minute
postFreqList = [len(frequencyMap[x]) for x in after_event_times]

# We'll have ticks every few minutes (more clutters the graph)
smallerXTicks = range(0, len(after_event_times), 5)
plt.xticks(smallerXTicks, [after_event_times[x] for x in smallerXTicks], rotation=90)

# Plot the post frequency
ax.plot(range(len(after_event_times)), postFreqList, color="blue", label="Posts")

ax.grid(b=True, which=u'major')
ax.legend()

ax.set_xlabel("Post Date")
ax.set_ylabel("Twitter Frequency")

plt.show()

temporally_relevant_tweets = [tweet 
                              for rel_time in after_event_times 
                              for tweet in frequencyMap[rel_time]]

print("Tweet Count:", len(temporally_relevant_tweets))

tokenizer = TweetTokenizer()
stops = stopwords.words("english")

# This list comprehension iterates through the tweet_list list, and for each
#  tweet, it tokenizes the tweet text
tokens = [
        token.lower() 
         for tweet in temporally_relevant_tweets 
             for token in tokenizer.tokenize(tweet["text"])
        ]
tokens = list(filter(lambda x: x not in stops and len(x) > 3, tokens))

print("Total Token Count:", len(tokens))
print("Unique Token Count:", len(set(tokens)))

tokens_freq = nltk.FreqDist(tokens)

print("\nFrequent Tokens:")
for token, count in tokens_freq.most_common(50):
    print(token, count)
    
    # This list comprehension iterates through the tweet_list list, and for each
#  tweet, it iterates through the hashtags list
htags = [
        hashtag["text"].lower() 
         for tweet in temporally_relevant_tweets 
             for hashtag in tweet["entities"]["hashtags"]
        ]

print("Total Hashtag Count:", len(htags))
print("Unique Hashtag Count:", len(set(htags)))

htags_freq = nltk.FreqDist(htags)

print("\nFrequent Hashtags:")
for tag, count in htags_freq.most_common(20):
    print(tag, count)
    
    # This list comprehension iterates through the tweet_list list, and for each
#  tweet, it iterates through the author list
authors = [tweet["user"]["screen_name"].lower() for tweet in temporally_relevant_tweets]

print("Total Author Count:", len(authors))
print("Unique Author Count:", len(set(authors)))

author_freq = nltk.FreqDist(authors)

print("\nActive Users:")
for author, count in author_freq.most_common(20):
    print(author, count)
    
    # This list comprehension iterates through the tweet_list list, and for each
#  tweet, it iterates through the URL list
urls = [
        url["expanded_url"]
         for tweet in temporally_relevant_tweets 
             for url in tweet["entities"]["urls"]
        ]

print("Total URL Count:", len(urls))
print("Unique URL Count:", len(set(urls)))

urls_freq = nltk.FreqDist(urls)

print("\nCommon URLs:")
for url, count in urls_freq.most_common(20):
    print(url, count)
    
    # Find retweets
retweets = list(filter(lambda x: "retweeted_status" in x, temporally_relevant_tweets))
print("Retweet Count:", len(retweets))

# But we also want to filter out retweets whose original tweet
#  was posted BEFORE our event
retweets = list(filter(
    lambda tweet: crisisTime <= datetime.datetime.strptime(tweet["retweeted_status"]["created_at"], timeFormat),
    retweets
))
print("Recent Retweet Count:", len(retweets))

# For each retweet, get its pair
rt_pairs = [(tweet["retweeted_status"]["id"], tweet["retweeted_status"]["retweet_count"]) 
          for tweet in retweets]

# Get the unique tweet IDs
uniq_ids = set([x[0] for x in rt_pairs])

# Map retweet IDs to their counts
rt_map = {}
for local_id in uniq_ids:
    max_rt_count = max([x[1] for x in filter(lambda x: x[0] == local_id, rt_pairs)])
    rt_map[local_id] = max_rt_count
    
# Sort the retweets by count
top_5_rts = sorted(rt_map, key=rt_map.get, reverse=True)[:5]

# For each of the top 5, print them
for rt in top_5_rts:
    
    tweet = list(filter(lambda x: x["retweeted_status"]["id"] == rt, retweets))[0]
    
    print(rt, rt_map[rt])
    print("\tTweet:", tweet["text"])
    print()
    
    # What keywords are we interested in?
targetKeywords = crisisInfo[selectedCrisis]["keywords"]

# Build an empty map for each keyword we are seaching for
targetCounts = {x:[] for x in targetKeywords}
totalCount = []

# For each minute, pull the tweet text and search for the keywords we want
for t in sortedTimes:
    timeObj = frequencyMap[t]
    
    # Temporary counter for this minute
    localTargetCounts = {x:0 for x in targetKeywords}
    localTotalCount = 0
    
    for tweetObj in timeObj:
        tweetString = tweetObj["text"].lower()

        localTotalCount += 1
        
        # Add to the counter if the target keyword is in this tweet
        for keyword in targetKeywords:
            if ( keyword in tweetString ):
                localTargetCounts[keyword] += 1
                
    # Add the counts for this minute to the main counter
    totalCount.append(localTotalCount)
    for keyword in targetKeywords:
        targetCounts[keyword].append(localTargetCounts[keyword])
        
# Now plot the total frequency and frequency of each keyword
fig, ax = plt.subplots()
fig.set_size_inches(11, 8.5)

# Set title
plt.title("Tweet Frequency")

# ticks interval
smallerXTicks = range(0, len(sortedTimes), 15)
plt.xticks(smallerXTicks, [sortedTimes[x] for x in smallerXTicks], rotation=90)

# Plot in log-scale because overall tweet volume is much higher than relevant volume
ax.semilogy(range(len(frequencyMap)), totalCount, label="Total")

# Plot a simple X where the disaster occurred
ax.scatter([crisisXCoord], [100], c="r", marker="x", s=100, label="Crisis")

# For each keyword in our target keywords...
for keyword in targetKeywords:
    
    # Print the target count for each keyword and time
    ax.semilogy(range(len(frequencyMap)), targetCounts[keyword], label=keyword)
    
# Legend and titles
ax.set_xlabel("Post Date")
ax.set_ylabel("Log(Frequency)")
ax.legend()
ax.grid(b=True, which=u'major')

plt.show()

relevant_keyword_tweets = {}

for rel_keyword in targetKeywords:
    relevant_tweets = [tweet 
                       for this_time in 
                           filter(lambda x: x >= crisisTime, sortedTimes) 
                       for tweet in 
                           filter(lambda y: rel_keyword in y["text"].lower(), 
                                  frequencyMap[this_time])
                      ]
    
    relevant_keyword_tweets[rel_keyword] = relevant_tweets
    print("Keyword:", rel_keyword, "Count:", len(relevant_tweets))
    
    for target_keyword, textually_relevant_tweets in relevant_keyword_tweets.items():

    print("\n------------------------")
    print("Keyword:", target_keyword, "Count:", len(textually_relevant_tweets))

    # This list comprehension iterates through the tweet_list list, and for each
    #  tweet, it iterates through the hashtags list
    htags = [
            hashtag["text"].lower() 
             for tweet in textually_relevant_tweets 
                 for hashtag in tweet["entities"]["hashtags"]
            ]

    print("Total Hashtag Count:", len(htags))
    print("Unique Hashtag Count:", len(set(htags)))

    htags_freq = nltk.FreqDist(htags)

    print("\nFrequent Hashtags:")
    for tag, count in htags_freq.most_common(20):
        print(tag, count)

    # This list comprehension iterates through the tweet_list list, and for each
    #  tweet, it iterates through the author list
    authors = [tweet["user"]["screen_name"].lower() for tweet in textually_relevant_tweets]

    print("\nTotal Author Count:", len(authors))
    print("Unique Author Count:", len(set(authors)))

    author_freq = nltk.FreqDist(authors)

    print("\nActive Users:")
    for author, count in author_freq.most_common(20):
        print(author, count)

    # This list comprehension iterates through the tweet_list list, and for each
    #  tweet, it iterates through the URL list
    urls = [
            url["expanded_url"]
             for tweet in textually_relevant_tweets 
                 for url in tweet["entities"]["urls"]
            ]

    print("Total URL Count:", len(urls))
    print("Unique URL Count:", len(set(urls)))

    urls_freq = nltk.FreqDist(urls)

    print("\nCommon URLs:")
    for url, count in urls_freq.most_common(20):
        print(url, count)
        
        all_rel_tweets = {tweet["id"]:tweet 
                  for local_tweet_list in relevant_keyword_tweets.values() 
                  for tweet in local_tweet_list}

print("Unique textually relevant tweets:", len(all_rel_tweets))

textually_relevant_tweets = list(all_rel_tweets.values())

# This list comprehension iterates through the tweet_list list, and for each
#  tweet, it tokenizes the tweet text
tokens = [
        token.lower() 
         for tweet in textually_relevant_tweets 
             for token in tokenizer.tokenize(tweet["text"])
        ]
tokens = list(filter(lambda x: x not in stops and len(x) > 3, tokens))

print("Total Token Count:", len(tokens))
print("Unique Token Count:", len(set(tokens)))

tokens_freq = nltk.FreqDist(tokens)

print("\nFrequent Tokens:")
for token, count in tokens_freq.most_common(20):
    print(token, count)

# This list comprehension iterates through the tweet_list list, and for each
#  tweet, it iterates through the hashtags list
htags = [
        hashtag["text"].lower() 
         for tweet in textually_relevant_tweets 
             for hashtag in tweet["entities"]["hashtags"]
        ]

print("\nTotal Hashtag Count:", len(htags))
print("Unique Hashtag Count:", len(set(htags)))

htags_freq = nltk.FreqDist(htags)

print("\nFrequent Hashtags:")
for tag, count in htags_freq.most_common(20):
    print(tag, count)

# This list comprehension iterates through the tweet_list list, and for each
#  tweet, it iterates through the author list
authors = [tweet["user"]["screen_name"].lower() for tweet in textually_relevant_tweets]

print("\nTotal Author Count:", len(authors))
print("Unique Author Count:", len(set(authors)))

author_freq = nltk.FreqDist(authors)

print("\nActive Users:")
for author, count in author_freq.most_common(20):
    print(author, count)

# This list comprehension iterates through the tweet_list list, and for each
#  tweet, it iterates through the URL list
urls = [
        url["expanded_url"]
         for tweet in textually_relevant_tweets 
             for url in tweet["entities"]["urls"]
        ]

print("\nTotal URL Count:", len(urls))
print("Unique URL Count:", len(set(urls)))

urls_freq = nltk.FreqDist(urls)

print("\nCommon URLs:")
for url, count in urls_freq.most_common(20):
    print(url, count)
    
    with open("/Users/yutingliao/Desktop/INST728 E/relevant_tweet_output_keywords_updated.json", "w") as out_file:
    for tweet in textually_relevant_tweets:
        out_file.write("%s\n" % json.dumps(tweet))
        
        