# 📌 TMDB Movies Dataset Visualization

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter

# Step 2: Load Dataset
df = pd.read_csv("tmdb_5000_movies.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Step 3: Data Preprocessing

# Parse JSON columns (genres, production_companies, production_countries, spoken_languages)
def parse_json_column(column):
    def parse(x):
        try:
            items = ast.literal_eval(x)
            return [i['name'] for i in items]
        except:
            return []
    return column.apply(parse)

df['genres_parsed'] = parse_json_column(df['genres'])

# Step 4: Basic statistics and info
print("\nDataset Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# Step 5: Visualizations

plt.figure(figsize=(10,6))
sns.histplot(df['budget'], bins=50, kde=False)
plt.title('Distribution of Movie Budgets')
plt.xlabel('Budget')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['revenue'], bins=50, kde=False)
plt.title('Distribution of Movie Revenues')
plt.xlabel('Revenue')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['popularity'], bins=50, kde=True)
plt.title('Distribution of Movie Popularity')
plt.xlabel('Popularity')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['vote_average'], bins=30, kde=True)
plt.title('Distribution of Movie Vote Average')
plt.xlabel('Vote Average')
plt.ylabel('Count')
plt.show()

# Scatter plot: Popularity vs Vote Average
plt.figure(figsize=(10,6))
sns.scatterplot(x='popularity', y='vote_average', data=df)
plt.title('Popularity vs Vote Average')
plt.xlabel('Popularity')
plt.ylabel('Vote Average')
plt.show()

# Bar plot: Top 10 genres by count
all_genres = sum(df['genres_parsed'], [])
genre_counts = Counter(all_genres)
top_genres = dict(genre_counts.most_common(10))

plt.figure(figsize=(10,6))
sns.barplot(x=list(top_genres.values()), y=list(top_genres.keys()), palette='viridis')
plt.title('Top 10 Movie Genres by Count')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Correlation heatmap of numeric features
numeric_cols = ['budget', 'popularity', 'revenue', 'vote_average', 'vote_count']
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.show()
