%matplotlib inline

import datetime
import json
import os

import numpy as np

# For plotting
import matplotlib
import matplotlib.pyplot as plt

# For mapping
from mpl_toolkits.basemap import Basemap

from nltk.tokenize import TweetTokenizer
#-77.119759,38.791645,-76.909393,38.995548

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

# Save only those tweets with tweet['coordinates']['coordinates'] entity
def coordinate_filter(tweet):
    return "coordinates" in tweet and tweet["coordinates"] != None

geo_tweets = list(filter(coordinate_filter, relevant_tweets))
geo_tweet_count = len(geo_tweets)

print ("Number of Geo Tweets:", geo_tweet_count)

# Save only those tweets with tweet['place'] entity
def place_filter(tweet):
    return "place" in tweet and tweet["place"] != None and tweet["coordinates"] == None

placed_tweets = list(filter(place_filter, relevant_tweets))
placed_tweet_count = len(placed_tweets)

print ("Number of Place Tweets:", placed_tweet_count)

# GPS-coded tweets vs. Place-coded tweets
print("GPS-coded Tweet:")
print(json.dumps(geo_tweets[0]["coordinates"], indent=2))
print(json.dumps(geo_tweets[0]["place"], indent=2))
print()

print("Place-coded Tweet:")
print(json.dumps(placed_tweets[0]["place"], indent=2))

# For each geo-coded tweet, extract its GPS coordinates
geoCoord = [x["coordinates"]["coordinates"] for x in geo_tweets]

# Now we build a map of the world using Basemap
land_color = 'lightgray'
water_color = 'lightblue'

# Create a nice, big figure
fig, ax = plt.subplots(figsize=(24,24))

# Build our map, focusing on most of the world and using
#  a Mercator project (many map projections exist)
worldMap = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=80,
                   llcrnrlon=-180, urcrnrlon=180, resolution='l')

# Make the map readable
worldMap.fillcontinents(color=land_color, lake_color=water_color, zorder=1)
worldMap.drawcoastlines()
worldMap.drawparallels(np.arange(-90.,120.,30.))
worldMap.drawmeridians(np.arange(0.,420.,60.))
worldMap.drawmapboundary(fill_color=water_color, zorder=0)
ax.set_title('World Tweets')

place_point = worldMap(
    crisisInfo[selectedCrisis]["place"][1], # Longitude
    crisisInfo[selectedCrisis]["place"][0], # Latitude
)
worldMap.scatter(place_point[0], place_point[1], 
                 s=1000, marker="o", color="blue", zorder=2,
                label="Disaster Point")

# Convert points from GPS coordinates to (x,y) coordinates
convPoints = [worldMap(p[0], p[1]) for p in geoCoord]

# Split out points for X,Y lists, which we'll use for our
#  standard Matplotlib plotting
x = [p[0] for p in convPoints]
y = [p[1] for p in convPoints]

# Plot the points on the map
worldMap.scatter(x, y, 
                 s=100, marker='x', color="red", zorder=2,
                label="GPS Tweets")

plt.legend()
plt.show()

# Get the bounding box for our crisis
bBox = crisisInfo[selectedCrisis]["box"]

fig, ax = plt.subplots(figsize=(11,8.5))

# Create a new map to hold the shape file data
targetMap = Basemap(llcrnrlon=bBox["lowerLeftLon"], 
                    llcrnrlat=bBox["lowerLeftLat"], 
                    urcrnrlon=bBox["upperRightLon"], 
                    urcrnrlat=bBox["upperRightLat"], 
                    projection='merc',
                    resolution='i', area_thresh=10000)

targetMap.fillcontinents(color=land_color, lake_color=water_color, 
                         zorder=1)
targetMap.drawcoastlines()
targetMap.drawstates()
targetMap.drawparallels(np.arange(-90.,120.,30.))
targetMap.drawmeridians(np.arange(0.,420.,60.))
targetMap.drawmapboundary(fill_color=water_color, zorder=0)
targetMap.drawcountries()

place_point = targetMap(
    crisisInfo[selectedCrisis]["place"][1], # Longitude
    crisisInfo[selectedCrisis]["place"][0], # Latitude
)
targetMap.scatter(place_point[0], place_point[1], 
                 s=100, marker="o", color="blue", zorder=2,
                label="Disaster Point")

# Now we build the polygon for filtering
# Convert from lon, lat of lower-left to x,y coordinates
llcCoord = targetMap(bBox["lowerLeftLon"], bBox["lowerLeftLat"])

# Same for upper-right corner
urcCoord = targetMap(bBox["upperRightLon"], bBox["upperRightLat"])

# Now make the polygon we'll us for filtering
boxPoints = np.array([[llcCoord[0], llcCoord[1]], 
                      [llcCoord[0], urcCoord[1]], 
                      [urcCoord[0], urcCoord[1]], 
                      [urcCoord[0], llcCoord[1]]])
boundingBox = matplotlib.patches.Polygon(boxPoints)

# For each geo-coded tweet, extract coordinates and convert 
# them to the Basemap space
convPoints = [targetMap(p[0], p[1]) for p in geoCoord]

# Track points within our bounding box
plottable = []

# For each point, check if it is within the bounding box or not
for point in convPoints:
    x = point[0]
    y = point[1]

    if ( boundingBox.contains_point((x, y))):
        plottable.append(point)

# Plot points in our target
targetMap.scatter([p[0] for p in plottable], [p[1] for p in plottable], s=100, 
                  marker='x', color="red", zorder=2)
            
print ("Tweets in Target Area:", len(plottable))
print ("Tweets outside:", (geo_tweet_count - len(plottable)))

plt.legend()
plt.show()

# This function takes a bounding box and finds its center point
#  NOTE: This is a not-so-great hack and can lead to strange behavior
#  (e.g., points in the middle of lakes or at random houses)
def flatten_bbox(tweet):
    lat = 0.0
    lon = 0.0
    
    p_count = 0
    for poly in tweet["place"]["bounding_box"]["coordinates"]:
        for p in poly:
            lat += p[1]
            lon += p[0]
            p_count += 1
        
    # Take the average location
    if ( p_count > 0 ):
        lat = lat / p_count
        lon = lon / p_count
        
    return (lon, lat)

# Extract flattened GPS coordinates
place_geocodes = [flatten_bbox(x) for x in placed_tweets]

# Now we build a map of the world using Basemap
land_color = 'lightgray'
water_color = 'lightblue'

# Create a nice, big figure
fig, ax = plt.subplots(figsize=(24,24))

# Build our map, focusing on most of the world and using
#  a Mercator project (many map projections exist)
worldMap = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=80,
                   llcrnrlon=-180, urcrnrlon=180, resolution='l')

# Make the map readable
worldMap.fillcontinents(color=land_color, lake_color=water_color, zorder=1)
worldMap.drawcoastlines()
worldMap.drawparallels(np.arange(-90.,120.,30.))
worldMap.drawmeridians(np.arange(0.,420.,60.))
worldMap.drawmapboundary(fill_color=water_color, zorder=0)
ax.set_title('Place-Coded Tweets')

place_point = worldMap(
    crisisInfo[selectedCrisis]["place"][1], # Longitude
    crisisInfo[selectedCrisis]["place"][0], # Latitude
)

worldMap.scatter(place_point[0], place_point[1], 
                 s=1000, marker="o", color="blue", zorder=2,
                label="Disaster Point")

# Convert points from GPS coordinates to (x,y) coordinates
convPoints = [worldMap(p[0], p[1]) for p in geoCoord]

# Split out points for X,Y lists, which we'll use for our
#  standard Matplotlib plotting
x = [p[0] for p in convPoints]
y = [p[1] for p in convPoints]

# Plot the points on the map
worldMap.scatter(x, y, 
                 s=100, marker='x', color="red", zorder=2,
                label="GPS Tweets")

# Place points in a different color
conv_place_points = [worldMap(p[0], p[1]) for p in place_geocodes]
# Plot the points on the map
worldMap.scatter([p[0] for p in conv_place_points], [p[1] for p in conv_place_points], 
                 s=100, marker='x', color="green", zorder=2,
                label="Placed Tweets")

plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(11,8.5))

# Create a new map to hold the shape file data
targetMap = Basemap(llcrnrlon=bBox["lowerLeftLon"], 
                    llcrnrlat=bBox["lowerLeftLat"], 
                    urcrnrlon=bBox["upperRightLon"], 
                    urcrnrlat=bBox["upperRightLat"], 
                    projection='merc',
                    resolution='i', area_thresh=10000)

targetMap.fillcontinents(color=land_color, lake_color=water_color, 
                         zorder=1)
targetMap.drawcoastlines()
targetMap.drawparallels(np.arange(-90.,120.,30.))
targetMap.drawmeridians(np.arange(0.,420.,60.))
targetMap.drawmapboundary(fill_color=water_color, zorder=0)
targetMap.drawcountries()

place_point = targetMap(
    crisisInfo[selectedCrisis]["place"][1], # Longitude
    crisisInfo[selectedCrisis]["place"][0], # Latitude
)
targetMap.scatter(place_point[0], place_point[1], 
                 s=100, marker="o", color="blue", zorder=2,
                label="Disaster Point")

# Now we build the polygon for filtering
# Convert from lon, lat of lower-left to x,y coordinates
llcCoord = targetMap(bBox["lowerLeftLon"], bBox["lowerLeftLat"])

# Same for upper-right corner
urcCoord = targetMap(bBox["upperRightLon"], bBox["upperRightLat"])

# Now make the polygon we'll us for filtering
boxPoints = np.array([[llcCoord[0], llcCoord[1]], 
                      [llcCoord[0], urcCoord[1]], 
                      [urcCoord[0], urcCoord[1]], 
                      [urcCoord[0], llcCoord[1]]])
boundingBox = matplotlib.patches.Polygon(boxPoints)

# For each geo-coded tweet, extract coordinates and convert 
# them to the Basemap space
convPoints = [targetMap(p[0], p[1]) for p in geoCoord]

# Track points within our bounding box
plottable = []

# For each point, check if it is within the bounding box or not
for point in convPoints:
    x = point[0]
    y = point[1]

    if ( boundingBox.contains_point((x, y))):
        plottable.append(point)

# Plot points in our target
targetMap.scatter([p[0] for p in plottable], [p[1] for p in plottable], s=100, 
                  marker='x', color="red", zorder=2)

print ("GPS Tweets in Target Area:", len(plottable))
print ("GPS Tweets outside:", (geo_tweet_count - len(plottable)))

# Place points in a different color
plottable_p2 = []
conv_place_points = [targetMap(p[0], p[1]) for p in place_geocodes]
for point in conv_place_points:
    x = point[0]
    y = point[1]

    if ( boundingBox.contains_point((x, y))):
        plottable_p2.append(point)
        
# Plot the points on the map
targetMap.scatter([p[0] for p in plottable_p2], [p[1] for p in plottable_p2], 
                 s=100, marker='x', color="green", zorder=2,
                label="Placed Tweets")
        
print ("Placed Tweets in Target Area:", len(plottable_p2))
print ("Placed Tweets outside:", (placed_tweet_count - len(plottable_p2)))

plt.legend()
plt.show()

# Now we build the polygon for filtering
llcCoord = (bBox["lowerLeftLon"], bBox["lowerLeftLat"])

# Same for upper-right corner
urcCoord = (bBox["upperRightLon"], bBox["upperRightLat"])

# Now make the polygon we'll us for filtering
boxPoints = np.array([[llcCoord[0], llcCoord[1]], 
                      [llcCoord[0], urcCoord[1]], 
                      [urcCoord[0], urcCoord[1]], 
                      [urcCoord[0], llcCoord[1]]])
boundingBox = matplotlib.patches.Polygon(boxPoints)

# Our list of spatially relevant tweets
spatial_relevant_tweets_gps = []
spatial_relevant_tweets_placed = []

# First check GPS tweets
for tweet in geo_tweets:
    x, y = tweet["coordinates"]["coordinates"]
    if ( boundingBox.contains_point((x, y))):
        spatial_relevant_tweets_gps.append(tweet)
        
print("Spatially Relevant GPS-coded Tweets:", len(spatial_relevant_tweets_gps))

# Then check Place tweets
for tweet in placed_tweets:
    x, y = flatten_bbox(tweet)
    if ( boundingBox.contains_point((x, y))):
        spatial_relevant_tweets_placed.append(tweet)
        
print("Spatially Relevant Tweets:", len(spatial_relevant_tweets_placed))

# Merge relevant tweets
spatial_relevant_tweets = spatial_relevant_tweets_gps + spatial_relevant_tweets_placed

# Print the relevant tweets
for tweet in spatial_relevant_tweets[:10]:
    print("Tweet By:", tweet["user"]["screen_name"])
    print("\t", "Tweet Text:", tweet["text"])
    print("\t", "Tweet Time:", tweet["created_at"])
    print("\t", "Source:", tweet["source"])
    print("\t", "Twitter's Guessed Language:", tweet["lang"])
    if ( "place" in tweet ):
        print("\t", "Tweet Location:", tweet["place"]["full_name"])
    print("-----")
    
    from IPython.display import display
from IPython.display import Image

geoTweetsWithMedia = list(filter(lambda tweet: "media" in tweet["entities"], 
                                 spatial_relevant_tweets))
print ("Tweets with Media:", len(geoTweetsWithMedia))

if ( len(geoTweetsWithMedia) == 0 ):
    print ("Sorry, not tweets with media...")

for tweet in geoTweetsWithMedia:
    print(tweet["text"])
    for media in tweet["entities"]["media"]:
        print("\tType:", media["type"])
        print("\t%s" % media["expanded_url"])
        display(Image(url=media["media_url"]))
        
        # Twitter's time format, for parsing the created_at date
timeFormat = "%a %b %d %H:%M:%S +0000 %Y"

# Frequency map for tweet-times
rel_frequency_map = {}
for tweet in relevant_tweets:
    # Parse time
    currentTime = datetime.datetime.strptime(tweet['created_at'], timeFormat)

    # Flatten this tweet's time
    currentTime = currentTime.replace(second=0,minute = 0)

    # If our frequency map already has this time, use it, otherwise add
    extended_list = rel_frequency_map.get(currentTime, [])
    extended_list.append(tweet)
    rel_frequency_map[currentTime] = extended_list

# Frequency map for tweet-times
geo_frequency_map = {}
for tweet in spatial_relevant_tweets:
    # Parse time
    currentTime = datetime.datetime.strptime(tweet['created_at'], timeFormat)

    # Flatten this tweet's time
    currentTime = currentTime.replace(second=0,minute = 0)

    # If our frequency map already has this time, use it, otherwise add
    extended_list = geo_frequency_map.get(currentTime, [])
    extended_list.append(tweet)
    geo_frequency_map[currentTime] = extended_list
    
    
# Fill in any gaps
times = sorted(set(rel_frequency_map.keys()).union(set(geo_frequency_map)))
firstTime = times[0]
lastTime = times[-1]
thisTime = firstTime

# We want to look at per-minute data, so we fill in any missing minutes
timeIntervalStep = datetime.timedelta(0, 60)    # Time step in seconds
while ( thisTime <= lastTime ):

    rel_frequency_map[thisTime] = rel_frequency_map.get(thisTime, [])
    geo_frequency_map[thisTime] = geo_frequency_map.get(thisTime, [])
        
    thisTime = thisTime + timeIntervalStep

# Count the number of minutes
print ("Start Time:", firstTime)
print ("Stop Time:", lastTime)
print ("Processed Times:", len(rel_frequency_map))

fig, ax = plt.subplots()
fig.set_size_inches(22, 17)

plt.title("Tweet Frequencies")

sortedTimes = sorted(rel_frequency_map.keys())
gpsFreqList = [len(geo_frequency_map[x]) for x in sortedTimes]
postFreqList = [len(rel_frequency_map[x]) for x in sortedTimes]

smallerXTicks = range(0, len(sortedTimes), 120)
plt.xticks(smallerXTicks, [sortedTimes[x] for x in smallerXTicks], rotation=90)

xData = range(len(sortedTimes))

ax.plot(xData, postFreqList, color="blue", label="Posts")
ax.plot(xData, gpsFreqList, color="green", label="GPS Posts")

ax.grid(b=True, which=u'major')
ax.legend()

plt.show()

