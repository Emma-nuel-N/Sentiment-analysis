# Sentiment-analysis
Overview
This code is designed to perform sentiment analysis on tourist accommodation reviews. It reads a CSV file containing reviews and their corresponding hotel/restaurant names, preprocesses the text data, calculates the sentiment polarity for each review, and categorizes the sentiment as positive, negative, or neutral. The code then visualizes the distribution of sentiments using bar plots and pie charts.

Requirements
Before running the code, ensure you have the following libraries installed:

pandas
seaborn
numpy
spacy
nltk
string
matplotlib
scikit-learn
textblob
wordcloud
You can install the required libraries using pip:

c
Copy code
pip install pandas seaborn numpy spacy nltk string matplotlib scikit-learn textblob wordcloud
Dataset
The code reads the data from the CSV file named 'tourist_accommodation_reviews.csv', which contains the following columns:

ID: Unique identifier for each review
Review Date: The date when the review was posted
Location: The location of the accommodation
Hotel/Restaurant name: The name of the hotel or restaurant
Review: The actual review text
Usage
Import the required libraries:
python
Copy code
import pandas as pd
import seaborn as sns
import numpy as np
import spacy
import nltk
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, metrics, roc_curve, auc, accuracy_score, classification_report, ConfusionMatrixDisplay
from nltk.stem.porter import PorterStemmer
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
Load the dataset and preprocess the reviews:
python
Copy code
df = pd.read_csv('tourist_accommodation_reviews.csv')
# Perform data preprocessing and filtering of reviews
# ...
Perform sentiment analysis and visualization:
python
Copy code
# Calculate sentiment polarity for each review
# ...
# Categorize sentiments and create visualizations
# ...
Display the results:
python
Copy code
# Show the results and visualizations
# ...
Data Preprocessing
The code performs the following data preprocessing steps on the review text:

Convert the text to lowercase.
Tokenize the text into words.
Remove stop words.
Perform stemming to reduce words to their base form.
Sentiment Analysis
The sentiment analysis is done using the TextBlob library, which calculates the polarity of the review text. The polarity ranges from -1 to 1, where -1 represents a negative sentiment, 1 represents a positive sentiment, and 0 represents a neutral sentiment.

Visualizations
The code creates visualizations to display the distribution of sentiments in the reviews. It includes a bar plot showing the count of each sentiment (positive, negative, neutral) and a pie chart displaying the percentage distribution of sentiments.

Note
This code is specifically designed for sentiment analysis on the given dataset of tourist accommodation reviews. If you want to use it for a different dataset or task, you may need to modify the data preprocessing steps and the sentiment analysis logic accordingly.


# libraries 
NLTK (Natural Language Tool Kit),
Textblob,
numpy


# Conclusion
The analysis has shown that Rawai View café & Bar have the least negative reviews hence is the most positively reviewed hotel/restaurant,
while Patong Seafood has the most negative reviews indicating that it is least liked out of the 30 hotels/restaurant. 

# Dataset
University of Salford
