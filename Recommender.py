import pandas as pd 
import numpy as np 

import re

import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import MinMaxScaler

import scipy.sparse as sp

from sklearn.metrics.pairwise import cosine_similarity


#Importing data 
df= pd.read_csv("C:/Users/ADMIN\Documents/My projects/Creator-Brand-Recommender/combined_instagram_data.csv")

print(df.columns)

#Shape of the data
print (f'Number of entries: {df.shape[0]}')
print (f'Number of columns: {df.shape[1]}')


# Check for duplicated rows
duplicated_count = df.duplicated().sum()

# Get detailed information about the data
df_info = df.info()

# Get statistical summary
df_description = df.describe()


print("### Duplicated Rows Count ###")
print(f'Total duplicated rows: {duplicated_count}\n')

print("### DataFrame Info ###")
df_info  
print("\n")

print("### DataFrame Statistical Summary ###")
print(df_description)


#checking for missing  values
df.isnull().sum()

# Data cleaning 

# Define keyword dictionaries for niches
niche_keywords = {
    'Health': ['health','fitness', 'workout', 'gym', 'health', 'trainer', 'yoga','fit','dermatologist','swimmer','pharmacy','hospital','clinic','surgery','wellness'],
    'Lifestyle':['lifestyle','mum','dad','mother','father','mama','marriage', 'married','therapy','spa','pet','model','wife','parent'],
    'Fashion': ['fashion', 'style', 'designer', 'clothing', 'beauty', 'makeup','clothes','dressing','wear','lotion','perfume','skin care','stylist'],
    'Travel': ['travel', 'wanderlust', 'vacation', 'adventure', 'explore','safari'],
    'Tech': ['tech', 'gadgets', 'software', 'developer', 'AI', 'programmer','machine','data'],
    'Food': ['food', 'recipe', 'chef', 'cooking', 'baking', 'restaurant','hotel','kitchen','foodie'],
    'Art': ['art', 'artist', 'drawing', 'painting', 'illustration', 'design','comedy','podcast','craft','photograph','choreography','actor','actress','writer','host','poet','storyteller','author'],
    'Music': ['music', 'singer', 'band', 'producer', 'songwriter', 'DJ','rapper','vocalist','saxophonist','musician'],
    'Business':['business','finance','financial','money','consultancy','shop','consulting','entrepreneur','store'],
}

# Preprocess the biography text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['biography_clean'] = df['biography'].fillna('').apply(clean_text)

# Match keywords to assign all matching niches
def assign_niches(bio, keywords_dict):
    niches = [niche for niche, keywords in keywords_dict.items() if any(keyword in bio for keyword in keywords)]
    return ' '.join(niches) if niches else 'Other'  # Join multiple niches with a space, or return 'Other'

# Apply the function to the clean biography column
df['niche'] = df['biography_clean'].apply(lambda x: assign_niches(x, niche_keywords))

# Save the updated dataset 
save_path = "C:/Users/ADMIN/Documents/My projects/Creator-Brand-Recommender/combined_instagram_data_with_niches.csv"
df.to_csv(save_path, index=False)
print(f"Dataset with niches saved successfully at: {save_path}")

print(df.shape)


# List of columns to drop
columns_to_drop = [
    'profile_name', 'full_name', 'fbid', 'id', 'external_url', 'business_category_name', 'posts',
    'profile_url', 'is_private', 'url', 'is_joined_recently', 'has_channel',
    'partner_id', 'timestamp','profile_image_link', 'input', 'source_file', 'processed_at'
]

# Drop the columns from the DataFrame
df = df.drop(columns=columns_to_drop)

print(df.shape)

## Remove columns with more than 50% missing data 
# Calculate the threshold for 50% missing data
threshold = len(df) * 0.5
df = df.dropna(axis=1, thresh=threshold)


print(df.shape)


## filling avg_engagement column 

# Set up the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Histogram 
sns.histplot(df['avg_engagement'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Histogram of avg_engagement')
axes[0].set_xlabel('Avg Engagement')
axes[0].set_ylabel('Frequency')

# Boxplot 
sns.boxplot(x=df['avg_engagement'], ax=axes[1], color='lightgreen')
axes[1].set_title('Boxplot of avg_engagement')
axes[1].set_xlabel('Avg Engagement')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

#Since the data is highly skewed, impute missing values with the median
median_val = df['avg_engagement'].median()
df['avg_engagement'].fillna(median_val, inplace=True)

#Followers column 
null_followers = df[df['followers'].isnull()]
print(null_followers.head())
#Most of it's entries in other columns are empty hence we delete it 
df= df.dropna(subset=['followers'])

#Drop category name since it was replaced by niche
df = df.drop(columns=['category_name'])

# Drop the 'biography' column since it's now represented by 'niche'
df= df.drop(columns=['biography'])
df= df.drop(columns=['biography_clean'])
df.shape

#checking for missing  values
df.isnull().sum()

#EDA

# Summary statistics for numeric columns
summary_stats = df[['followers', 'posts_count', 'avg_engagement', 'following', 'highlights_count']].describe()

summary_stats


fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot histograms
sns.histplot(df['followers'], ax=axes[0, 0], kde=True).set_title('Followers Distribution')
sns.histplot(df['posts_count'], ax=axes[0, 1], kde=True).set_title('Posts Count Distribution')
sns.histplot(df['avg_engagement'], ax=axes[0, 2], kde=True).set_title('Average Engagement Distribution')
sns.histplot(df['following'], ax=axes[1, 0], kde=True).set_title('Following Distribution')
sns.histplot(df['highlights_count'], ax=axes[1, 1], kde=True).set_title('Highlights Count Distribution')

# Plot boxplots
sns.boxplot(x=df['followers'], ax=axes[0, 0])
sns.boxplot(x=df['posts_count'], ax=axes[0, 1])
sns.boxplot(x=df['avg_engagement'], ax=axes[0, 2])
sns.boxplot(x=df['following'], ax=axes[1, 0])
sns.boxplot(x=df['highlights_count'], ax=axes[1, 1])

plt.tight_layout()
plt.show()

# Correlation matrix
correlation_matrix = df[['followers', 'posts_count', 'avg_engagement', 'following', 'highlights_count']].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Heatmap')
plt.show()


sns.countplot(x='is_business_account', data=df, ax=axes[0, 0]).set_title('Business Account Distribution')
sns.countplot(x='is_professional_account', data=df, ax=axes[0, 1]).set_title('Professional Account Distribution')
sns.countplot(x='is_verified', data=df, ax=axes[1, 0]).set_title('Verified Account Distribution')

plt.tight_layout()
plt.show()

#Data pre-processing 

#Transfrom niche column to numeric 
Vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = Vectorizer.fit_transform(df['niche'])

numeric_features = [
    'followers', 'posts_count', 'is_business_account', 
    'is_professional_account', 'is_verified', 
    'avg_engagement', 'following', 'highlights_count'
]

# Normalize 
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[numeric_features])

normalized_df = pd.DataFrame(normalized_data, columns=numeric_features)

# Combine all the features 
numeric_sparse = sp.csr_matrix(normalized_data)
combined_features = sp.hstack([tfidf_matrix, numeric_sparse])

# Content based recommender 

# Compute similarity
similarity_matrix = cosine_similarity(combined_features)

def recommend_accounts(account_name, df, similarity_matrix, top_n=5):
    # Get the index of the account
    account_idx = df[df['account'] == account_name].index[0]
    
    # Fetch similarity scores for the account
    similarity_scores = list(enumerate(similarity_matrix[account_idx]))
    
    # Sort by similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of top_n most similar accounts
    similar_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    
    # Return the names of the similar accounts
    return df.iloc[similar_indices]['account']
  
  #Testing
recommendations = recommend_accounts('kate_actress', df, similarity_matrix, top_n=5)

print("Recommended accounts:")
for idx, account in enumerate(recommendations, 1): 
    print(f"{idx}. {account}")

