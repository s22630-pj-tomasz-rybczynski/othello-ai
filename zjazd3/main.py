"""
Authors: Tomasz Rybczyński, Filip Marcoń

Precautions:
- Python 3.8
- numpy
- json

This script provides a movie recommendation system based on user ratings of movies. It contains a MovieRecommender class
that can be used to calculate the Euclidean distance between two users based on their movie ratings, and recommend movies
to a user based on their ratings of other movies.
"""

import numpy as np
import json

class MovieRecommender:
    """
    A class for recommending movies to users based on their ratings of other movies.

    Attributes:
    -----------
    data : dict
        A dictionary containing user ratings for movies.
    """

    def __init__(self, data):
        """
        Initializes a MovieRecommender object.

        Parameters:
        -----------
        data : dict
            A dictionary containing user ratings for movies.
        """
        self.data = data

    def euclidean_score(self, user1, user2):
        """
        Calculates the Euclidean distance between two users based on their movie ratings.

        Parameters:
        -----------
        user1 : str
            The name of the first user.
        user2 : str
            The name of the second user.

        Returns:
        --------
        float
            The Euclidean distance between the two users.
        """
        common_movies = {movie for movie in self.data[user1] if movie in self.data[user2]}
        if len(common_movies) == 0:
            return 0
        squared_diff = sum([pow(self.data[user1][movie] - self.data[user2][movie], 2) for movie in common_movies])
        return 1 / (1 + np.sqrt(squared_diff))

    def recommend(self, user):
        """
        Recommends movies to a user based on their ratings of other movies.

        Parameters:
        -----------
        user : str
            The name of the user to recommend movies to.

        Returns:
        --------
        tuple
            A tuple containing two lists of movies: the top 5 recommended movies and the bottom 5 recommended movies.
        """
        scores = [(self.euclidean_score(user, other_user), other_user) for other_user in self.data if other_user != user]
        scores.sort()
        scores.reverse()
        scores = scores[0:5]

        recom_movies = {}
        for score, other_user in scores:
            for movie in self.data[other_user]:
                if movie not in self.data[user] or self.data[user][movie] == 0:
                    if movie not in recom_movies:
                        recom_movies[movie] = (self.data[other_user][movie], score)
                    else:
                        current_score = recom_movies[movie][1]
                        recom_movies[movie] = (self.data[other_user][movie], current_score + score)

        movie_list = [(score, movie) for movie, (rating, score) in recom_movies.items()]
        movie_list.sort()
        movie_list.reverse()

        top_5_movies = movie_list[:5]
        bottom_5_movies = movie_list[-5:]

        return top_5_movies, bottom_5_movies

# Load data from json file
with open('data.json') as f:
    data = json.load(f)

recommender = MovieRecommender(data)

# Get user's full name as input
user_full_name = input("Enter your full name: ")

top_5_movies, bottom_5_movies = recommender.recommend(user_full_name)

print("Top 5 movie recommendations:")
for _, movie in top_5_movies:
    print(movie)

print("\nBottom 5 movie recommendations:")
for _, movie in bottom_5_movies:
    print(movie)
