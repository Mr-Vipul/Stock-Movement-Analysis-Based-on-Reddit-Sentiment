from src.scraper import fetch_reddit_data
from src.sentiment_analysis import process_data
from src.model import prepare_data, train_multiple_models, evaluate_models, select_best_model, predict_movement

def main():
    import os
    os.makedirs("data", exist_ok=True)

    # Query user for the stock name
    print("Welcome to the Stock Movement Analysis Tool!")
    stock_name = input("Enter a stock name to analyze (e.g., 'Tata Motors', 'Sensex'): ").strip()
    
    if not stock_name:
        print("No stock name provided. Exiting...")
        return

    # Fetch Reddit data for the entered stock
    subreddit_name = "IndianStockMarket"  # Default subreddit for stock discussions
    print(f"Fetching data from r/{subreddit_name} for the stock: '{stock_name}'...")
    try:
        data = fetch_reddit_data(subreddit_name, stock_name, limit=500)  # Fetching Reddit data for the stock
        if data.empty:
            print(f"No data found for the stock '{stock_name}'. Try again with a different stock.")
            return
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Save raw data to CSV
    raw_file = "data/raw_reddit_data.csv"
    data.to_csv(raw_file, index=False)
    print(f"Raw data saved to {raw_file}.")

    # Process data (perform sentiment analysis)
    processed_file = "data/processed_reddit_data.csv"
    print("Processing data (sentiment analysis)...")
    try:
        processed_data = process_data(raw_file, processed_file)
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    print(f"Processed data saved to {processed_file}.")

    # Train and evaluate models
    print("Training and evaluating models...")
    try:
        X_train, X_test, y_train, y_test = prepare_data(processed_file)
        trained_models = train_multiple_models(X_train, y_train)
        model_scores = evaluate_models(trained_models, X_test, y_test)
        best_model = select_best_model(model_scores, trained_models)
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        return

    # Predict stock movement based on the fetched data
    print(f"Predicting stock movement for '{stock_name}'...")
    latest_score = processed_data['sentiment'].iloc[-1]  # Use the sentiment of the latest Reddit post
    num_comments = processed_data['num_comments'].iloc[-1]  # Use the number of comments for the latest post

    # Make a prediction using the best model
    movement = predict_movement(best_model, latest_score, num_comments)
    print(f"Predicted Stock Movement for '{stock_name}': {movement}")

if __name__ == "__main__":
    main()
