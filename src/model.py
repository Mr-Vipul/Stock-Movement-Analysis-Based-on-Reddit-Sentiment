import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from imblearn.over_sampling import RandomOverSampler

def prepare_data(input_file):
    df = pd.read_csv(input_file)
    
    # Ensure the sentiment column is numeric
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 1 if x > 0 else 0)
    
    X = df[['score', 'num_comments']]  # Features: You can add more features if necessary
    y = df['sentiment_label']  # Target variable (positive/negative sentiment)
    
    # Apply oversampling to handle class imbalance
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier

def train_multiple_models(X_train, y_train):
    """
    Train multiple models and return the trained models.
    :param X_train: Training features.
    :param y_train: Training labels.
    :return: Dictionary of trained models.
    """
    # Define models with class weights to handle class imbalance
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=45, class_weight='balanced'),
        'LogisticRegression': LogisticRegression(random_state=45, class_weight='balanced'),
        'SVM': SVC(random_state=45, class_weight='balanced')
    }
    
    trained_models = {}
    
    # Train each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models


from sklearn.metrics import classification_report, accuracy_score

def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate multiple models and print their performance.
    
    :param trained_models: Dictionary of trained models.
    :param X_test: Test features.
    :param y_test: Test labels.
    :return: Dictionary with accuracy scores for each model.
    """
    model_scores = {}
    
    for name, model in trained_models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of {name}: {accuracy:.2f}")
        # Added zero_division parameter to handle cases where precision is ill-defined
        print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred, zero_division=0)}")
        model_scores[name] = accuracy
    
    return model_scores


def select_best_model(model_scores, trained_models):
    """
    Select the best-performing model based on accuracy.
    
    :param model_scores: Dictionary of model accuracies.
    :param trained_models: Dictionary of trained models.
    :return: The best-performing model.
    """
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = trained_models[best_model_name]
    print(f"Best model: {best_model_name} with accuracy {model_scores[best_model_name]:.4f}")
    
    return best_model

def predict_movement(model, sentiment_score, num_comments):
    """
    Predict stock movement based on sentiment and number of comments.
    
    :param model: Trained model.
    :param sentiment_score: Sentiment score of the latest Reddit post.
    :param num_comments: Number of comments on the latest Reddit post.
    :return: Predicted stock movement ('Buy', 'Sell', 'Hold').
    """
    input_data = pd.DataFrame({
        'score': [sentiment_score],
        'num_comments': [num_comments]
    })
    
    prediction = model.predict(input_data)
    return "Buy" if prediction == 1 else "Sell" if prediction == 0 else "Hold"
