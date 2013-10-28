# File: test_collab_rec.py

import unittest
import os
import collab_rec as cr
from math import sqrt
from collections import Counter

guideToDataMiningUsers = {
  "Angelica": Counter({"Blues Traveler": 3.5, "Broken Bells": 2.0,
          "Norah Jones": 4.5, "Phoenix": 5.0,
          "Slightly Stoopid": 1.5,
          "The Strokes": 2.5, "Vampire Weekend": 2.0}),
   
 "Bill": Counter({"Blues Traveler": 2.0, "Broken Bells": 3.5,
     "Deadmau5": 4.0, "Phoenix": 2.0,
     "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0}),
 
 "Chan": Counter({"Blues Traveler": 5.0, "Broken Bells": 1.0,
      "Deadmau5": 1.0, "Norah Jones": 3.0, "Phoenix": 5,
      "Slightly Stoopid": 1.0}),
 
 "Dan": Counter({"Blues Traveler": 3.0, "Broken Bells": 4.0,
     "Deadmau5": 4.5, "Phoenix": 3.0,
     "Slightly Stoopid": 4.5, "The Strokes": 4.0,
     "Vampire Weekend": 2.0}),
 
 "Hailey": Counter({"Broken Bells": 4.0, "Deadmau5": 1.0,
      "Norah Jones": 4.0, "The Strokes": 4.0,
      "Vampire Weekend": 1.0}),
 
 "Jordyn": Counter({"Broken Bells": 4.5, "Deadmau5": 4.0,
       "Norah Jones": 5.0, "Phoenix": 5.0,
       "Slightly Stoopid": 4.5, "The Strokes": 4.0,
       "Vampire Weekend": 4.0}),
 
 "Sam": Counter({"Blues Traveler": 5.0, "Broken Bells": 2.0,
     "Norah Jones": 3.0, "Phoenix": 5.0,
     "Slightly Stoopid": 4.0, "The Strokes": 5.0}),
 
 "Veronica": Counter({"Blues Traveler": 3.0, "Norah Jones": 5.0,
        "Phoenix": 4.0, "Slightly Stoopid": 2.5,
        "The Strokes": 3.0})
}

class TestDataImportFunctions(unittest.TestCase):

  tempDataFilename = 'data/unittest_temp_data.txt'

  def setUp(self):
    self.users = cr.importUserPlays('data/practice_data_50.txt')
    cr.saveData(self.users, self.tempDataFilename)

  def test_importUserPlays(self):
    self.assertEqual(14, len(self.users[2]))
    self.assertEqual(1, self.users[65][6])
    self.assertEqual(1, self.users[65][423])
    self.assertFalse(8 in self.users[65])

  def test_saveData_loadData(self):
    self.assertEqual(self.users,
      cr.loadData(self.tempDataFilename))

  def tearDown(self):
    os.remove(self.tempDataFilename)


class TestDistanceMetrics(unittest.TestCase):

  def setUp(self):
    # Own data
    self.users = cr.importUserPlays('data/practice_data_50.txt')
    self.alan = {
      'Blues Traveler': 1.0,
      'Phoenix': 4.0
    }
    # Data from http://guidetodatamining.com/guide/ch2/DataMining-ch2.pdf
    self.clara = {
      'Blues Traveler': 4.75,
      'Norah Jones': 4.5,
      'Phoenix': 5.0,
      'The Strokes': 4.25,
      'Weird Al': 4.0
    }
    self.robert = {
      'Blues Traveler': 4.0,
      'Norah Jones': 3.0,
      'Phoenix': 5.0,
      'The Strokes': 2.0,
      'Weird Al': 1.0
    }
    self.otherUsers = guideToDataMiningUsers

  def test_pearsonCorrelation(self):
    self.assertAlmostEqual(1.0, cr.pearsonCorrelation(self.clara, self.robert))

  def test_manhattanDistance(self):
    self.assertAlmostEqual(4.0, cr.manhattanDistance(self.alan, self.robert))

  def test_euclideanDistance(self):
    self.assertAlmostEqual(sqrt(18.5), cr.euclideanDistance(
      self.otherUsers['Angelica'], self.otherUsers['Bill']))

  def test_cosineSimilarity(self):
    self.assertAlmostEqual(70.0/(sqrt(101.875)*sqrt(55)),
      cr.cosineSimilarity(self.clara, self.robert))

  def test_numMutuallyScoredItems(self):
    self.assertEqual(5, cr.numMutuallyScoredItems(self.clara, self.robert))
    self.assertEqual(4, cr.numMutuallyScoredItems(
      self.otherUsers['Dan'], self.otherUsers['Hailey']))
    self.assertEqual(2, cr.numMutuallyScoredItems(self.alan, self.robert))


class TestRecommenderClass(unittest.TestCase):

  def setUp(self):
    self.manhattanR = cr.Recommender(guideToDataMiningUsers,
        cr.manhattanDistance, False, 7)
    self.euclideanR = cr.Recommender(guideToDataMiningUsers,
        cr.euclideanDistance, False, 4)
    self.pearsonR = cr.Recommender(guideToDataMiningUsers,
        cr.pearsonCorrelation, True, 6)
    self.cosineR = cr.Recommender(guideToDataMiningUsers,
        cr.cosineSimilarity, True, 6)

  def test_nearestNeighbors(self):
    self.assertEqual(
      [('Veronica', 2.0), ('Sam', 4.0), ('Chan', 4.0), ('Dan', 4.5),
      ('Angelica', 5.0), ('Bill', 5.5), ('Jordyn', 7.5)],
      self.manhattanR.nearestNeighbors('Hailey')
    )

  def test_basic_recommend(self):
    self.assertEqual('Blues Traveler',
                      self.manhattanR.recommend('Jordyn',1)[0][0])
    self.assertEqual('Blues Traveler',
                      self.euclideanR.recommend('Jordyn',1)[0][0])
    self.assertEqual('Blues Traveler',
                      self.pearsonR.recommend('Jordyn',1)[0][0])
    self.assertEqual('Blues Traveler',
                      self.cosineR.recommend('Jordyn',1)[0][0])


if __name__ == '__main__':
  unittest.main()