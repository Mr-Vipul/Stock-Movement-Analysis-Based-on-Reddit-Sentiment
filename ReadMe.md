# Stock Movement Analysis Based on Reddit Sentiment

This repository contains a Python-based tool to predict stock movement using sentiment analysis of Reddit discussions. The project uses data scraped from Reddit, sentiment analysis using Hugging Face Transformers, and machine learning models to predict whether a stock is likely to move positively or negatively.

## Project Features
1. Scraper: Fetches Reddit posts about a specific stock.
2. Sentiment Analysis: Analyzes the sentiment of Reddit posts using a pre-trained Hugging Face pipeline.
3. Machine Learning: Trains multiple models (Random Forest, Logistic Regression, and SVM) to predict stock movement based on sentiment and other features.
4. Prediction: Predicts stock movement (Buy/Sell) based on recent Reddit discussions.

## Setup Instructions
### Prerequisites
1. Python 3.7 or higher
2. pip for installing dependencies


### Dependencies
Install the required Python packages:
```python
pip install -r requirements.txt
```

### File Structure
```python
.
├── data/                   # Folder for storing raw and processed data
├── src/                    # Source code folder
│   ├── scraper.py          # Fetches Reddit data
│   ├── sentiment_analysis.py # Performs sentiment analysis
│   ├── model.py            # Contains data preprocessing, model training, and evaluation
├── main.py                 # Main script for running the pipeline
├── README.md               # Project overview and setup instructions
├── requirements.txt        # List of required Python packages
└── report.pdf              # Detailed project report
```

### How to Run the Project
1. Clone the repository:
```
git clone https://github.com/Mr-Vipul/Stock-Movement-Analysis-Based-on-Reddit-Sentiment.git
cd Stock-Movement-Analysis-Based-on-Reddit-Sentiment
```

2.Update Reddit API credentials in ```src/scraper.py```:

```
REDDIT_CLIENT_ID = "Your_reddit_client_id"
REDDIT_CLIENT_SECRET = "Your_reddit_client_secret"
REDDIT_USER_AGENT = "python:stock_analysis_project:v1.0.0 (by /u/Your_reddit_username)"
```

3.Run the pipeline:
```
python main.py
```
4.Enter the stock name when prompted, e.g., Tata Motors.
 
5.View the predicted stock movement and metrics.


## License

[MIT](https://choosealicense.com/licenses/mit/)
