import numpy as np

def predict_with_model(model, X):
    try:
        preds = model.predict(X)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = None
        return preds, proba
    except Exception as e:
        print(f"Error making prediction: {e}")
        return np.zeros(len(X)), None
