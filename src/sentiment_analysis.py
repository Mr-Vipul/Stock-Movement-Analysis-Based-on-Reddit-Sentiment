import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def process_data(input_file, output_file):
    """
    Perform sentiment analysis on the text data.

    :param input_file: Path to the raw data CSV.
    :param output_file: Path to save the processed data.
    :return: Processed DataFrame.
    """
    # Load the input data
    df = pd.read_csv(input_file)

    # Initialize the Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    # Perform sentiment analysis on the 'title' column
    df['sentiment'] = df['title'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    # Save the processed data to the output file
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

    # Return the processed DataFrame
    return df

