import joblib
from sklearn.metrics import classification_report


def save_model(model, path):
    """Save trained model"""
    joblib.dump(model, path)


def evaluate_model(y_true, y_pred):
    """Print classification report"""
    print(classification_report(y_true, y_pred))