#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

df = pd.read_csv('df_train.csv')

df.head()


# In[21]:


# df Preprocessing

# Fill missing values
df['following'] = df['following'].fillna(0)
df['followers'] = df['followers'].fillna(0)
df['actions'] = df['actions'].fillna(0)
df['is_retweet'] = df['is_retweet'].fillna(0)

df['tweet_length'] = df['Tweet'].apply(len)  # Length of the tweet
df['hashtag_count'] = df['Tweet'].apply(lambda x: x.count('#'))  # Number of hashtags
df['mention_count'] = df['Tweet'].apply(lambda x: x.count('@'))  # Number of mentions
df['url_count'] = df['Tweet'].apply(lambda x: x.count('http'))  # Number of URLs
df['capitalized_count'] = df['Tweet'].apply(lambda x: sum(1 for c in x if c.isupper()))  # Capitalized words
df['exclamation_count'] = df['Tweet'].apply(lambda x: x.count('!'))  # Exclamation symbols
df['question_mark_count'] = df['Tweet'].apply(lambda x: x.count('?'))  # Question marks


# Encode the target variable (Type)
df['Type'] = df['Type'].map({'Spam': 1, 'Quality': 0})
print(df.head())


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
# Convert the 'Tweet' col to string and handle non-string entries
df['Tweet'] = df['Tweet'].astype(str)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#\w+", "", text)     # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = " ".join([word for word in text.split() if word not in STOPWORDS])  # Remove stopwords
    return text

df['cleaned_tweet'] = df['Tweet'].astype(str).apply(clean_text)
vectorizer = TfidfVectorizer(max_features=500)  
X_text_feature = vectorizer.fit_transform(df['cleaned_tweet']).toarray()  # Convert to array

# Re-extract features now that all entries are strings
df['tweet_length'] = df['Tweet'].apply(len)  # Length of the tweet
df['hashtag_count'] = df['Tweet'].apply(lambda x: x.count('#'))  
df['mention_count'] = df['Tweet'].apply(lambda x: x.count('@'))  
df['url_count'] = df['Tweet'].apply(lambda x: x.count('http'))  
df['capitalized_count'] = df['Tweet'].apply(lambda x: sum(1 for c in x if c.isupper())) 
df['exclamation_count'] = df['Tweet'].apply(lambda x: x.count('!'))
df['question_mark_count'] = df['Tweet'].apply(lambda x: x.count('?'))  

print(df.head(9))
X_meta_features = df[['following', 'followers', 'actions', 'is_retweet', 'tweet_length', 
                      'hashtag_count', 'mention_count', 'url_count', 
                      'capitalized_count', 'exclamation_count', 'question_mark_count']]

# Step 3: Combine both sets of features (NLP + Metaheuristic)
import numpy as np
X_combined = np.hstack((X_text_feature, X_meta_features.values))
y = df['Type']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


# Display the shape of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:




