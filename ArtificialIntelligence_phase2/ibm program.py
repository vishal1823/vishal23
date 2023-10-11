import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Download the VADER lexicon (only need to do this once)
nltk.download('vader_lexicon')

# Sample marketing text data
data = {
    'Text': [
        "Amazon Puma",
        "Myntra Ajio ",
        "Adidas Amazon woodland ",
        "Ajio Adidas Ajio Reliance Bruton ",
        "Flipkart Puma ",
        "Reliance Ajio Puma ",
        "Puma Reliance Ajio woodland ",
        "woodland Myntra Flipkart ",
    ]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Create a SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# Calculate sentiment scores for each text in the DataFrame
df['Sentiment Scores'] = df['Text'].apply(lambda x: sia.polarity_scores(x))

# Extract compound scores and sentiment labels
df['Compound'] = df['Sentiment Scores'].apply(lambda x: x['compound'])
df['Sentiment'] = df['Compound'].apply(lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral'))

# Calculate word frequencies
words = " ".join(df['Text']).split()
word_freq = Counter(words)

# Visualize sentiment distribution and word cloud
plt.figure(figsize=(16, 8))

# Sentiment Distribution
plt.subplot(2, 2, 1)
sns.countplot(data=df, x='Sentiment', palette='Set2')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')

# Sentiment Distribution Pie Chart
plt.subplot(2, 2, 2)
sentiment_stats = df['Sentiment'].value_counts(normalize=True).reset_index()
sentiment_stats.columns = ['Sentiment', 'Percentage']
plt.pie(sentiment_stats['Percentage'], labels=sentiment_stats['Sentiment'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightblue'])
plt.title('Sentiment Distribution (Percentage)')

# Word Cloud
plt.subplot(2, 2, 3)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Most Frequent Brands/products')

# Word Frequency Bar Chart (Top 10 words)
plt.subplot(2, 2, 4)
common_words = word_freq.most_common(10)
common_words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
sns.barplot(data=common_words_df, x='Frequency', y='Word', palette='Set2')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.title('Top 10 Most Frequent Brands/products')

plt.tight_layout()
plt.show()

# Display detailed statistics
print("\nSentiment Statistics:")
print(sentiment_stats)

# Explore specific examples of each sentiment
positive_example = df[df['Sentiment'] == 'Positive']['Text'].iloc[0]
negative_example = df[df['Sentiment'] == 'Negative']['Text'].iloc[0]
neutral_example = df[df['Sentiment'] == 'Neutral']['Text'].iloc[0]

print("\nExample of Positive Sentiment:")
print(positive_example)

print("\nExample of Negative Sentiment:")
print(negative_example)

print("\nExample of Neutral Sentiment:")
print(neutral_example)
