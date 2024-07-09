# News Recommender System

This project implements a sophisticated news recommender system using Python, focusing on three different approaches: Content-Based Filtering, Collaborative Filtering, and a Hybrid Recommender System.
Overview

The objective of this project was to create a personalized news recommendation engine using a dataset of approximately 1950 articles scraped from various news sources. Here’s a breakdown of the approaches implemented:
## Content-Based Filtering

Content-Based Filtering recommends articles similar to a given article by analyzing their content and computing similarity scores using word embeddings. It focuses on the attributes of the articles themselves to find similarities.
## Collaborative Filtering

Collaborative Filtering generates recommendations based on user interactions and preferences. In this project, user behavior was simulated by creating user profiles with random topic preferences. Articles were recommended based on these preferences, with click probabilities and ratings assigned to simulate real-world interactions.
## Hybrid Recommender System

The Hybrid Recommender System combines the strengths of both Content-Based and Collaborative Filtering. It integrates content-based similarity scores with collaborative filtering rankings to provide more accurate and personalized recommendations.
## Key Features

    Content-Based Filtering: Recommends articles based on article content and similarity scores.
    Collaborative Filtering: Recommends articles based on simulated user preferences and interactions.
    Hybrid Recommender System: Combines content-based and collaborative filtering for enhanced recommendations.
    Dataset: Utilizes a dataset of approximately 1950 news articles scraped from various sources.

## Usage

To run the recommender system:

    Clone the repository.
    Install the necessary dependencies using pip install -r requirements.txt.

## Conclusion

This project demonstrates the implementation and comparison of different recommendation techniques—Content-Based Filtering, Collaborative Filtering, and Hybrid Recommender System. It showcases their individual strengths and the combined capabilities in delivering personalized news content to users.