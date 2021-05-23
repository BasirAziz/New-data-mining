%matplotlib inline

import json
# For our first piece of code, we need to import the package 
# that connects to Reddit. Praw is a thin wrapper around reddit's 
# web APIs and works well

import praw

# Now we specify a "unique" user agent for our code
# This is primarily for identification, I think, and some
# user-agents of bad actors might be blocked
redditApi = praw.Reddit(client_id='OdpBKZ1utVJw8Q',
                        client_secret='KH5zzauulUBG45W-XYeAS5a2EdA',
                        user_agent='crisis_informatics_v01')

subreddit = "worldnews"

targetSub = redditApi.subreddit(subreddit)

submissions = targetSub.new(limit=10)
for post in submissions:
    print(post.title)
    
    subreddit = "worldnews"

targetSub = redditApi.subreddit(subreddit)

submissions = targetSub.hot(limit=5)
for post in submissions:
    print(post.title)
    
    subreddit = "worldnews+news"

targetSub = redditApi.subreddit(subreddit)
submissions = targetSub.new(limit=10)
for post in submissions:
    print(post.title, post.author)
    
    subreddit = "worldnews"

breadthCommentCount = 5

targetSub = redditApi.subreddit(subreddit)

submissions = targetSub.hot(limit=1)

for post in submissions:
    print (post.title)
    
    post.comment_limit = breadthCommentCount
    
    # Get the top few comments
    for comment in post.comments.list():
        if isinstance(comment, praw.models.MoreComments):
            continue
        
        print ("---", comment.name, "---")
        print ("\t", comment.body)
        
        for reply in comment.replies.list():
            if isinstance(reply, praw.models.MoreComments):
                continue
            
            print ("\t", "---", reply.name, "---")
            print ("\t\t", reply.body)

# As before, the first thing we do is import the Facebook
# wrapper

import facebook

fbAccessToken = "EAACEdEose0cBAK2kyW5pcrgzUUMqmr4uR1ppwlz1lC5aIhJyVLm9Bfo1jOXBQwILsVzlt28dSmqwPdX9DQQDLz5zMEZC3ZB6HYTj5LyZA5hKoa3YneQpRyg3cCxwmb0Ea6uazjxaJX2QLNkL7i6BTVhy0bZCZBfvVb29AFZARFXhjcmsFO8QhY2EEhFyZBXIucZD"

# Connect to the graph API, note we use version 2.7
graph = facebook.GraphAPI(access_token=fbAccessToken, version='2.7')

# What page to look at?
targetPage = "nytimes"

# Other options for pages:
# nytimes, bbc, bbcamerica, bbcafrica, redcross, disaster

maxPosts = 10 # How many posts should we pull?
maxComments = 5 # How many comments for each post?

post = graph.get_object(id=targetPage + '/feed')

# For each post, print its message content and its ID
for v in post["data"][:maxPosts]:
    print ("---")
    print (v["message"], v["id"])
        
    # For each comment on this post, print its number, 
    # the name of the author, and the message content
    print ("Comments:")
    comments = graph.get_object(id='%s/comments' % v["id"])
    for (i, comment) in enumerate(comments["data"][:maxComments]):
        print ("\t", i, comment["from"]["name"], comment["message"])
        
        # For our first piece of code, we need to import the package 
# that connects to Twitter. Tweepy is a popular and fully featured
# implementation.

import tweepy

# Use the strings from your Twitter app webpage to populate these four 
# variables. Be sure and put the strings BETWEEN the quotation marks
# to make it a valid Python string.

consumer_key = "1jFG5MF4PNf8zhg8Nmkk3kWVb"
consumer_secret = "MOfU9zxDvsk7nKHLnYvpTUeWW5C7PsXrS9TuwnvYcx3ANzc5LG"
access_token = "2343077714-N9yB6UKYegygTgTPl7xgm7PfUbLhO6TzqitlFP0"
access_secret = "d44DHDHV3CYmeuDnWbITumnPcnrVJwS0mJhgITQYWyXdx"

# Now we use the configured authentication information to connect
# to Twitter's API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

print("Connected to Twitter!")

# Get tweets from our timeline
public_tweets = api.home_timeline()

# print the first five authors and tweet texts
for tweet in public_tweets[:5]:
    print (tweet.author.screen_name, tweet.author.name, "said:", tweet.text)
    
    # Our search string
queryString = "earthquake"

# Perform the search
matchingTweets = api.search(queryString)

print ("Searched for:", queryString)
print ("Number found:", len(matchingTweets))

# For each tweet that matches our query, print the author and text
print ("\nTweets:")
for tweet in matchingTweets:
    print (tweet.author.screen_name, tweet.author.name, tweet.text)
    
    # Lets find only media or links about earthquakes
queryString = "earthquake (filter:media OR filter:links)"

# Perform the search
matchingTweets = api.search(queryString)

print ("Searched for:", queryString)
print ("Number found:", len(matchingTweets))

# For each tweet that matches our query, print the author and text
print ("\nTweets:")
for tweet in matchingTweets:
    print (tweet.author.screen_name, tweet.text)
    
    # Lets find only media or links about earthquakes
queryString = "earthquake (filter:media OR filter:links)"

# How many tweets should we fetch? Upper limit is 1,500
maxToReturn = 100

# Perform the search, and for each tweet that matches our query, 
# print the author and text
print ("\nTweets:")
for status in tweepy.Cursor(api.search, q=queryString).items(maxToReturn):
    print (status.author.screen_name, status.text)
    
    # First, we need to create our own listener for the stream
# that will stop after a few tweets
class LocalStreamListener(tweepy.StreamListener):
    """A simple stream listener that breaks out after X tweets"""
    
    # Max number of tweets
    maxTweetCount = 10
    
    # Set current counter
    def __init__(self):
        tweepy.StreamListener.__init__(self)
        self.currentTweetCount = 0
        
        # For writing out to a file
        self.filePtr = None
        
    # Create a log file
    def set_log_file(self, newFile):
        if ( self.filePtr ):
            self.filePtr.close()
            
        self.filePtr = newFile
        
    # Close log file
    def close_log_file(self):
        if ( self.filePtr ):
            self.filePtr.close()
    
    # Pass data up to parent then check if we should stop
    def on_data(self, data):

        print (self.currentTweetCount)
        
        tweepy.StreamListener.on_data(self, data)
            
        if ( self.currentTweetCount >= self.maxTweetCount ):
            return False

    # Increment the number of statuses we've seen
    def on_status(self, status):
        self.currentTweetCount += 1
        
        # Could write this status to a file instead of to the console
        print (status.text)
        
        # If we have specified a file, write to it
        if ( self.filePtr ):
            self.filePtr.write("%s\n" % status._json)
        
    # Error handling below here
    def on_exception(self, exc):
        print (exc)

    def on_limit(self, track):
        """Called when a limitation notice arrives"""
        print ("Limit", track)
        return

    def on_error(self, status_code):
        """Called when a non-200 status code is returned"""
        print ("Error:", status_code)
        return False

    def on_timeout(self):
        """Called when stream connection times out"""
        print ("Timeout")
        return

    def on_disconnect(self, notice):
        """Called when twitter sends a disconnect notice
        """
        print ("Disconnect:", notice)
        return

    def on_warning(self, notice):
        print ("Warning:", notice)
        """Called when a disconnection warning message arrives"""
        
        listener = LocalStreamListener()
localStream = tweepy.Stream(api.auth, listener)

# Stream based on keywords
localStream.filter(track=['earthquake', 'disaster'])

listener = LocalStreamListener()
localStream = tweepy.Stream(api.auth, listener)

# List of screen names to track
screenNames = ['bbcbreaking', 'CNews', 'bbc', 'nytimes']

# Twitter stream uses user IDs instead of names
# so we must convert
userIds = []
for sn in screenNames:
    user = api.get_user(sn)
    userIds.append(user.id_str)

# Stream based on users
localStream.filter(follow=userIds)

listener = LocalStreamListener()
localStream = tweepy.Stream(api.auth, listener)

# Specify coordinates for a bounding box around area of interest
# In this case, we use San Francisco
swCornerLat = 36.8
swCornerLon = -122.75
neCornerLat = 37.8
neCornerLon = -121.75

boxArray = [swCornerLon, swCornerLat, neCornerLon, neCornerLat]

# Say we want to write these tweets to a file
listener.set_log_file(open("tweet_log.json", "w"))

# Stream based on location
localStream.filter(locations=boxArray)

# Close the log file
listener.close_log_file()

