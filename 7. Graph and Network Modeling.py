%matplotlib inline

import datetime
import json
import string
import os

import numpy as np

# For plotting
import matplotlib.pyplot as plt

# Network analysis
import networkx as nx

import nltk # Used for FreqDist

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

# We'll use a directed graph since mentions/retweets are directional
graph = nx.DiGraph()
    
for tweet in relevant_tweets:
    userName = tweet["user"]["screen_name"].lower()
    graph.add_node(userName)

    mentionList = tweet["entities"]["user_mentions"]

    for otherUser in mentionList:
        otherUserName = otherUser["screen_name"].lower()
        if ( graph.has_node(otherUserName) == False ):
            graph.add_node(otherUserName)
            
        if ( graph.has_edge(userName, otherUserName)):
            graph[userName][otherUserName]["weight"] += 1
        else:
            graph.add_edge(userName, otherUserName, weight=1)
        
print ("Number of Users:", len(graph.node))


for edge in graph.edges():
    if ( graph[edge[0]][edge[1]]["weight"] > 1 ):
        print(edge, graph[edge[0]][edge[1]])
        
        # Now we prune for performance reasons
# remove all nodes with few edges

for i in range(5):
    nodeList = [n for n,d in graph.degree() if d<2]
    
    if ( len(nodeList) == 0 ):
        break
    
    print("Nodes to Delete:", len(nodeList))
    
    graph.remove_nodes_from(nodeList)
    print ("Number of Remaining Users:", len(graph.node))
    
    
    
# THis may take a while
pageRankList = nx.pagerank_numpy(graph)


highRankNodes = sorted(pageRankList.keys(), key=pageRankList.get, reverse=True)
for x in highRankNodes[:20]:
    print (x, pageRankList[x])
    
    
    #plt.hist([x for x in pageRankList.values()])
plt.plot(range(len(pageRankList)), sorted([x for x in pageRankList.values()]))

plt.grid()
plt.show()


plt.figure(figsize=(8,8))
pos = nx.spring_layout(graph, scale=200, iterations=100, k=0.2)
#pos = nx.fruchterman_reingold_layout(graph, weight="weight", iterations=100)
# pos = nx.random_layout(graph)
nx.draw(graph, 
        pos, 
        node_color='#A0CBE2', 
        width=0.7, 
        with_labels=False,
        node_size=50)

# Get the highest ranking nodes...
hrNames = highRankNodes[:10]

# Get a list of scores for these high-ranked nodes
scores = pageRankList.values()
min_val = min(scores)
max_val = max(scores)
hrValues = [((pageRankList[x]-min_val) / max_val) for x in hrNames]

# Draw our high-rank nodes with a larger size and different color
nx.draw_networkx_nodes(graph, pos, nodelist=hrNames,
                       node_size=200,
                       node_color=hrValues,
                       cmap=plt.cm.winter)

# Dummy dictionary that maps usernames to themselves
#  (we'll use this to set node labels)
hrDict = dict(zip(hrNames, hrNames))

# Add labels, so we can see them
nx.draw_networkx_labels(graph,
                        pos,
                        labels=hrDict,
                        fontsize=36,
                        font_color="g")

plt.axis('off')
plt.show()

from networkx.algorithms import community # Community analysis functions


# Use Girvan-Newman algorithm to find top-level community structure
community_iter = community.girvan_newman(graph)

# The first set of communities is the top level. Subsequent elements
#  in this iterator describe subgroups within communities. We'll 
#  only use level 1 for now.
top_level_comms = next(community_iter)


def draw_graph(graph):
    """
    Function for drawing a given graph using the spring layout
    algorithm.
    """
    
    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(graph, scale=200, iterations=100, k=0.2)
    # pos = nx.fruchterman_reingold_layout(graph, weight="weight", iterations=100)
    # pos = nx.random_layout(graph)
    nx.draw(subg, 
            pos, 
            node_color='#A0CBE2', 
            width=1, 
            with_labels=False,
            node_size=50)

    plt.axis('off')
    plt.show()
    
    
def find_auth_nodes(graph, limit=5):
    """
    Given a NetworkX Graph structure, use PageRank to find the most
    authoritative nodes in the graph.
    """
    
    # THis may take a while
    local_pg_rank = nx.pagerank_numpy(graph)
    
    # Rank the users by their PageRank score, and reverse the list
    #  so we can get the top users in the front of the list
    local_auths = sorted(local_pg_rank.keys(), key=local_pg_rank.get, reverse=True)
    
    # Take only the first few users
    local_targets = local_auths[:limit]

    # Print user name and PageRank score
    print("\tTop Users:")
    for x in local_targets:
        print ("\t", x, local_pg_rank[x])
        
    # In case we want to use these usernames later
    return local_targets

def user_hashtags(user_list, tweet_list, limit=5):
    """
    Simple function that finds all tweets by a given set of users,
    and prints the top few most frequent hashtags
    """
    
    # Keep only tweets authored by someone in our user set
    target_tweets = filter(
        lambda tweet: tweet["user"]["screen_name"].lower() in user_list, tweet_list)
    
    # This list comprehension iterates through the tweet_list list, and for each
    #  tweet, it iterates through the hashtags list
    htags = [
            hashtag["text"].lower() 
             for tweet in target_tweets 
                 for hashtag in tweet["entities"]["hashtags"]
            ]

    htags_freq = nltk.FreqDist(htags)

    print("\tFrequent Hashtags:")
    for tag, count in htags_freq.most_common(limit):
        print("\t", tag, count)

# Iterate through the communities and trim ones of smallish size
for i, comm in enumerate(top_level_comms):
    
    # We'll skip small communities
    if ( len(comm) < 10 ):
        continue
        
    print("Community: %d" % (i+1))
    print("\tUser Count: %d" % len(comm))
    
    # Use the username set produced by our community generator to 
    #  create a subgraph of only these users and the connections
    #  between them.
    subg = graph.subgraph(comm)
    
    # Given the subgraph...
    #  find the most authoritative nodes,
    find_auth_nodes(subg)
    
    #  the most frequent hashtags, and
    user_hashtags(comm, relevant_tweets, limit=10)
    
    #  then visualize the network
    draw_graph(subg)



