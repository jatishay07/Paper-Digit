import pickle
import numpy as np
from config import MODEL_WEIGHTS_PATH
from train import make_predictions, forward_prop, get_predictions, get_accuracy, load_data

with open(MODEL_WEIGHTS_PATH, "rb") as f:
    W1, b1, W2, b2 = pickle.load(f)

print("Loaded weights successfully from:", MODEL_WEIGHTS_PATH)

X_train, Y_train, X_dev, Y_dev = load_data()

_, _, _, A2 = forward_prop(W1, b1, W2, b2, X_dev)
predictions = get_predictions(A2)

acc = get_accuracy(predictions, Y_dev)
print(f"Model loaded correctly. Validation accuracy: {acc:.4f}")

print("Sample predictions:  ", predictions[:10])
print("Sample actual labels:", Y_dev[:10])