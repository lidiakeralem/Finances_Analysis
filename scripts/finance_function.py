import pandas as pd
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
