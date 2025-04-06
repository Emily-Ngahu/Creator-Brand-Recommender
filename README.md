# Creator-Brand-Recommender
![image](https://github.com/Emily-Ngahu/Creator-Brand-Recommender/blob/main/Brand_creator_connect.png)
## Project Aim

This project focuses on building a recommendation system to match Instagram creators with relevant brands based on their niche and engagement metrics. By leveraging Natural Language Processing (NLP) and machine learning techniques, the system identifies similar accounts to facilitate brand collaborations.

This recommender system aims to:
- Enhance Brand-Creator Matching: Helping brands find the most suitable influencers for partnerships.
- Optimize Engagement Insights: Providing data-driven insights into influencer performance.
- Automate Recommendations: Reducing manual efforts in searching for potential collaborations.

## Tools Used
- Python (Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn)

## Data

The dataset used for this project was collected from Instagram and contains various attributes of creator accounts, such as:
- Account Information: Username, profile name, biography, verification status.
- Engagement Metrics: Followers, posts count, average engagement rate.
- Business Indicators: Whether the account is a business or professional account.
- Niche Identification: Extracted from the biography text using keyword matching.

### Data Description

After data cleaning, the final dataset consists of 2,531 entries with 11 key features, including:
1. account - Instagram account name.
2. followers - Number of followers.
3. posts_count - Total number of posts.
4. is_business_account - Whether the account is a business profile.
5. is_professional_account - Whether the account is classified as professional.
6. is_verified - Whether the account has a verified badge.
7. avg_engagement - The average engagement rate per post.
8. following - Number of accounts followed.
9. highlights_count - Number of Instagram highlights.
10. niche - The categorized niche based on biography content.

## Data Cleaning
1. Removed irrelevant columns (e.g., profile URLs, timestamps, source files).
2. Handled missing values by replacing engagement metrics with median values.
3. Dropped columns with more than 50% missing data.
4. Extracted niches using keyword-matching from biographies.

## Data Insights
1. Verified accounts tend to have lower engagement despite having higher follower counts.
2. Business accounts generally have a higher average engagement rate than personal accounts.
3. Certain niches (e.g., fitness, fashion) demonstrate strong engagement correlations.

## Exploratory Data Analysis (EDA)

### 1. Univariate Analysis

Distribution of followers and posts count.

Average engagement rate per account type (business vs. personal).

Proportion of verified vs. non-verified accounts.

Most common niches among Instagram creators.

### 2. Multivariate Analysis

Correlation between engagement rate and follower count.

Relationship between posts count and engagement levels.

Comparison of engagement between different niches.

Model: Content-Based Recommender System

## Data Preprocessing

1. Feature Extraction: Combined TF-IDF vectorized biography text with normalized numerical engagement metrics.

2. Feature Normalization: Used MinMaxScaler to scale numerical data.

3. Similarity Calculation

 - Cosine Similarity was used to determine account similarity based on features.

4. Recommendation Function

- Input: Instagram account name.

- Output: Top 5 most similar accounts based on niche and engagement patterns.

Example Output

recommendations = recommend_accounts('kate_actress', df, similarity_matrix, top_n=5)
print(recommendations)

Output:

1. kie_kie__
2. pamela.ashley.uba
3. roco_runs
4. eddiebutita
5. mattrife

Model Performance

The recommender system successfully identifies influencers with similar engagement and niche categories. Future iterations could refine recommendations using deep learning models.
