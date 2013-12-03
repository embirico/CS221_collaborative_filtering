# File: collab_rec.py

# Note: much of the |Recommender| class code is based off of an example on
# http://guidetodatamining.com/

import pickle # Loading and saving of complex data types
#TODO consider using cpickle instead
from collections import defaultdict # Sparse dict representation
from collections import Counter # Sparse vector representation
from math import sqrt
import numpy as np
import time


# Data import export helpers ----------------------------------------
def importUserPlays(filename):
	"""Get dict of Counters of how much a user has played each game."""
	userPlays = {}
	with open(filename, 'r') as f:
		while True:
			line = f.readline()
			if not line: break # EOF
			line = line.strip()
			if not line: continue # We can ignore empty lines
			userId = int(line)
			userPlays[userId] = Counter()
			f.readline() # skip line 'likes'
			numLikes = int(f.readline())
			f.readline() # skip blank line
			for _ in range(numLikes):
				f.readline() # skip all likes
			f.readline() # skip blank line
			f.readline() # skip line 'dislikes'
			numDislikes = int(f.readline())
			f.readline() # skip blank line
			for _ in range(numDislikes):
				f.readline() # skip all dislikes
			f.readline() # skip blank line
			f.readline() # skip line 'plays'
			numPlayedGames = int(f.readline())
			f.readline() # skip blank line
			for _ in range(numPlayedGames):
				gameId, numPlays = f.readline().split()
				userPlays[userId][int(gameId)] = int(numPlays)
	return userPlays

def importUserPlaysAndLikesComposite(filename,
	playPoints=1, likePoints=10):
	"""
	Get dict of Counters of how a user rates a game.
	Each play of a game counts as |playPoints| points.
	Each like of a game counts as |likePoints| points.
	Each dislike of a game counts as -|likePoints| points.
	"""
	userRatings = {}
	with open(filename, 'r') as f:
		while True:
			line = f.readline()
			if not line: break # EOF
			line = line.strip()
			if not line: continue # We can ignore empty lines
			userId = int(line)
			userRatings[userId] = Counter()
			f.readline() # skip line 'likes'
			numLikes = int(f.readline())
			f.readline() # skip blank line
			for _ in range(numLikes):
				gameId = f.readline()
				userRatings[userId][int(gameId)] += likePoints
			f.readline() # skip blank line
			f.readline() # skip line 'dislikes'
			numDislikes = int(f.readline())
			f.readline() # skip blank line
			for _ in range(numDislikes):
				gameId = f.readline()
				userRatings[userId][int(gameId)] -= likePoints
			f.readline() # skip blank line
			f.readline() # skip line 'plays'
			numPlayedGames = int(f.readline())
			f.readline() # skip blank line
			for _ in range(numPlayedGames):
				gameId, numPlays = f.readline().split()
				userRatings[userId][int(gameId)] += int(numPlays) * playPoints
	return userRatings

def importDataRatio(filename):
	"""Get dict of Counters of percent of time a user has played each game."""
	# TODO
	pass

def saveData(data, filename):
	"""Save |data| object to |filename|."""
	with open(filename, 'w') as f:
		pickle.dump(data, f)

def loadData(filename):
	"""Returns the object saved with |saveData()|."""
	with open(filename, 'r') as f:
		return pickle.load(f)


# User to user distance metrics -------------------------------------
def pearsonCorrelation(user1, user2):
	"""
	Returns the Pearson Correlation Coefficient btw 2 users.

	|user1|, |user2| are Counters where keys are itemIds and values are scores.
	Code taken from http://guidetodatamining.com with minimal modification.
	TODO see if this actually works
	Used with Recommender, set lowDistMeansDissimilar=True.
	"""
	sum_xy = 0
	sum_x = 0
	sum_y = 0
	sum_x2 = 0
	sum_y2 = 0
	n = 0
	for key in user1:
	  if key in user2:
	    n += 1
	    x = user1[key]
	    y = user2[key]
	    sum_xy += x * y
	    sum_x += x
	    sum_y += y
	    sum_x2 += pow(x, 2)
	    sum_y2 += pow(y, 2)
	if n == 0:
	  return 0
	# now compute denominator
	denominator = (sqrt(sum_x2 - pow(sum_x, 2) / n)
	         * sqrt(sum_y2 - pow(sum_y, 2) / n))
	if denominator == 0:
		return 0
	else:
		return (sum_xy - (sum_x * sum_y) / n) / denominator

def manhattanDistance(user1, user2):
	"""
	Returns the Manhattan distance btw 2 users.

	|user1|, |user2| are Counters where keys are itemIds and values are scores.
	Code taken from http://guidetodatamining.com with minimal modification.
	Used with Recommender, set lowDistMeansDissimilar=False.
	"""
	dist = 0
	for itemId in user1:
		if itemId in user2:
			dist += abs(user1[itemId] - user2[itemId]) 
	return dist 


def euclideanDistance(user1, user2):
	"""
	Returns the Euclidean distance btw 2 users.

	|user1|, |user2| are Counters where keys are itemIds and values are scores.
	Used with Recommender, set lowDistMeansDissimilar=False.
	"""
	distSquared = 0
	for itemId in user1:
		if itemId in user2:
			distSquared += (user1[itemId] - user2[itemId])**2
	return sqrt(distSquared)

def cosineSimilarity(user1, user2):
	"""
	Returns the cosine similarity btw 2 users.

	|user1|, |user2| are Counters where keys are itemIds and values are scores.
	Used with Recommender, set lowDistMeansDissimilar=True.
	"""
	allKeys = set(user1) | set(user2)
	vec1 = [user1[key] for key in allKeys]
	vec2 = [user2[key] for key in allKeys]
	return float(np.dot(vec1, vec2)) \
					/ np.linalg.norm(vec1) / np.linalg.norm(vec2)

def numMutuallyScoredItems(user1, user2):
	"""
	Returns the number of items that both users have scored.

	|user1|, |user2| are Counters where keys are itemIds and values are scores.
	Used with Recommender, set lowDistMeansDissimilar=True.
	"""
	return len(set(user1.keys()) & set(user2.keys()))

def sumCommonScore(user1, user2):
	"""
	Returns the sum of minimum scores for items both users have rated.

	Used with Recommender, set lowDistMeansDissimilar=True.
	"""
	correlation = 0.0
	for itemId in user1:
		if itemId in user2:
			correlation += min(user1[itemId], user2[itemId])
	return correlation


# Recommender class -------------------------------------------------
class Recommender(object):
	"""
	TODO docstring

	Code inspired from http://guidetodatamining.com.
	"""
	def __init__(self, users, distanceMetric, lowDistMeansDissimilar, k):
		"""
		|users|: dict of dicts mapping users to their score for each item
		|distanceMetric|: function to compute distances btw user profiles
		|higherDistMeansDissimilar|: False if distanceMetric returning a low
		value means 2 profiles are similar (e.g. Manhattan Distance). True
		if distanceMetric returning a low value menas 2 profiles are dissimilar
		(e.g. Pearson Correlation)
		|k|: how many k-nearest neighbors to use in calculation
		"""
		self.users = users
		self.distFn = distanceMetric
		self.lowDistMeansDissimilar = lowDistMeansDissimilar
		self.k = k

	def nearestNeighbors(self, tgtUserId):
		"""
		Returns list of |self.k| (userId, dist) tuples sorted by
		descending similarity from |tgtUserId|
		"""
		usersAndDists = []
		for userId in self.users:
		  if userId != tgtUserId:
				distance = self.distFn(self.users[tgtUserId], self.users[userId])
				usersAndDists.append((userId, distance))
		usersAndDists.sort(key=lambda (_, dist): dist,
											reverse=self.lowDistMeansDissimilar)
		return usersAndDists[:self.k]

	def recommend(self, tgtUserId, nRecs=-1):
		"""
		Recommend |nRecs| |itemId|s for |tgtUserId|.

		For the moment actually returns a list of (itemId, score)
		to help with evaluating performance.
		"""
		# Setup
		recsDict = defaultdict(float) # default val will be 0.0
		tgtUserRatings = self.users[tgtUserId]
		# Get k nearest neighbors
		nearestNbrsAndDists = self.nearestNeighbors(tgtUserId)
		# Compute |totalDist| to be used later to weigh recs
		if self.lowDistMeansDissimilar:
			totalDist = sum([dist for (_, dist) in nearestNbrsAndDists])
			if totalDist == 0: totalDist = 1
		# Accumulate ratings for k nearest neighbors
		for nbrId, dist in nearestNbrsAndDists:
			# Create a measure so that similar users' recs count more
			weight = float(dist) / totalDist if self.lowDistMeansDissimilar \
							 else 1 #TODO do something smart here
							 				#to weigh manhattan or euclidean dists
			nbrRatings = self.users[nbrId]
			for itemId in nbrRatings:
				if True:
				# if itemId not in tgtUserRatings: # TODO put back
					recsDict[itemId] += nbrRatings[itemId] * weight
		# Get sorted list of recs
		recsDistsList = sorted(list(recsDict.items()),
													key=lambda (_, dist): dist,
													reverse=self.lowDistMeansDissimilar)
		return recsDistsList if nRecs < 0 else recsDistsList[:nRecs]

	def recommendForEveryUser(self, nRecs):
		"""
		Returns a dict of no more than |nRecs| recommendations for each user
		in the dataset. (Scores not included)
		"""
		return {user : zip(*self.recommend(user, nRecs))[0] for user in self.users}
		

# Mini helpers ------------------------------------------------------
def getGameInfo():
	gameInfoDict = loadData('data/game_info.pickle')
	gameInfo = defaultdict(lambda: ('??', '??'))
	for x in gameInfoDict:
		gameInfo[x] = gameInfoDict[x]
	return gameInfo

def testRecommenderForUser(tgtUserId, rec, userRatings, gameInfo):
	print 'USER {} HAS RATED...'.format(tgtUserId)
	for gameId in userRatings[tgtUserId]:
		print '{} : {} : {}'.format(gameInfo[gameId][0], gameInfo[gameId][1],
			userRatings[tgtUserId][gameId])
	print
	print 'AND WE RECOMMEND...'
	for gameId, _ in rec.recommend(tgtUserId, 6):
		print '{} : {}'.format(gameInfo[gameId][0], gameInfo[gameId][1])
	print
	print

def test1():
	gameInfo = getGameInfo()
	userRatings = importUserPlaysAndLikesComposite('data/practice_data_50.txt')
	rec = Recommender(userRatings, cosineSimilarity, True, 6)
	testRecommenderForUser(5, rec, userRatings, gameInfo)
	testRecommenderForUser(20, rec, userRatings, gameInfo)
	testRecommenderForUser(28, rec, userRatings, gameInfo)
	testRecommenderForUser(31, rec, userRatings, gameInfo)

def test2():
	gameInfo = getGameInfo()
	userRatings = importUserPlaysAndLikesComposite('data/practice_data_6623.txt')
	rec = Recommender(userRatings, numMutuallyScoredItems, True, 6)
	testRecommenderForUser(29410, rec, userRatings, gameInfo)
	testRecommenderForUser(8675, rec, userRatings, gameInfo)
	testRecommenderForUser(27452, rec, userRatings, gameInfo)
	testRecommenderForUser(33790, rec, userRatings, gameInfo)

def testCachingOnLargeFile():
	t = time.time()
	userRatings = importUserPlaysAndLikesComposite('data/practice_data_6623.txt')
	print 'Importing txt data took {}s'.format(time.time() - t)
	rec = Recommender(userRatings, sumCommonScore, True, 6)
	t = time.time()
	recsCache = rec.recommendForEveryUser(6)
	print 'Creating cache object took {}s'.format(time.time() - t)
	t = time.time()
	saveData(recsCache, 'data/recs_cache.pickle')
	print 'Saving recs_cache.pickle took {}s'.format(time.time() - t)

def testCacheLoading():
	t = time.time()
	cache = loadData('data/recs_cache.pickle')
	print 'Loading data took {}s'.format(time.time() - t)


# Main script -------------------------------------------------------
if __name__ == '__main__':
	testCachingOnLargeFile()
	# test2()



