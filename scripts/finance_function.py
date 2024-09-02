import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

# headline length
def calculate_headline_lengths(data):
    """Calculate basic statistics for headline lengths."""
    data['headline_length'] = data['headline'].apply(len)
    return data['headline_length'].describe()

# article per publisher
def count_articles_per_publisher(data):
    """Count the number of articles per publisher."""
    return data['publisher'].value_counts()

# data analysis
def analyze_publication_dates(data):
    """Analyze publication dates to identify trends over time."""
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    data = data.dropna(subset=['date'])  # Drop rows where date conversion failed
    data.set_index('date', inplace=True)
    daily_counts = data.resample('D').size()
    return daily_counts



# Optional: Function to visualize the data
import matplotlib.pyplot as plt

def plot_daily_counts(daily_counts):
    """Plot daily counts of articles."""
    plt.figure(figsize=(12, 6))
    daily_counts.plot()
    plt.title('Number of Articles Published Per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.show()


## sentiment analysis
def perform_sentiment_analysis(data):
    """Perform sentiment analysis on the headlines."""
    # Ensure 'headline' column exists
    if 'headline' not in data.columns:
        raise ValueError("Data must contain a 'headline' column.")

    # Function to analyze sentiment
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        # Classify the sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'

    # Apply sentiment analysis
    data['sentiment'] = data['headline'].apply(analyze_sentiment)
    sentiment_counts = data['sentiment'].value_counts()
    
    return data, sentiment_counts

## topic modeling
def perform_topic_modeling(data, n_topics=5):
    """Perform topic modeling on the headlines."""
    # Ensure 'headline' column exists
    if 'headline' not in data.columns:
        raise ValueError("Data must contain a 'headline' column.")

    # Vectorize the headlines
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['headline'])
    
    # Perform LDA topic modeling
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(X)
    
    # Get the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx}: {' '.join(top_words)}")

    return topics