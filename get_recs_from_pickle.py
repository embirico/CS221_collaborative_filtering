# File: get_recs_from_pickle.py
"""
A simple script to by run by the server to get recommendations that were
precomputed by collab_rec.py
"""

DESCRIPTION_STRING = """
Prints the number of recs for the user, followed by the (up to 6)
recommended gameIds, separated by spaces:

n gameId1 ... gameIdN
"""

import pickle
import argparse

def printRecsForUser(cache, userId):
	if userId not in cache:
		print '0'
	else:
		userRecs = cache[userId]
		print len(userRecs), ' '.join([str(gameId) for gameId in userRecs])

def main(args):
	cache = pickle.load(open(args.pickle_file, 'r'))
	printRecsForUser(cache, args.user_id)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=DESCRIPTION_STRING)
	parser.add_argument('pickle_file', type=str,
		help='Pickle file from collab_rec.py')
	parser.add_argument('user_id', type=int,
		help='User id to recommend for')
	args = parser.parse_args()
	main(args)