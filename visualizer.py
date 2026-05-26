#visualizer.py
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

def plot_visualizer(true_vals, pred_vals, categories, title="Fold Predictions"):
    """
    Plot true vs predicted values with category‑based markers.
    
    Parameters
    ----------
    true_vals : array-like
        Ground truth regression values for the fold.
    pred_vals : array-like
        Model predictions for the fold.
    categories : array-like
        Category labels aligned with true_vals (e.g., Good/Bad/Explode).
    title : str
        Plot title.
    """

    true_vals = np.array(true_vals).ravel()
    pred_vals = np.array(pred_vals).ravel()
    categories = np.array(categories)

    sample_idx = np.arange(len(true_vals))

    # Masks
    mask_good    = categories == "Good"
    mask_bad     = categories == "Bad"
    mask_explode = categories == "Explode"

    fig = go.Figure()

    # GOOD
    fig.add_trace(go.Scatter(
        x=sample_idx[mask_good],
        y=true_vals[mask_good],
        mode="markers",
        name="Good (True)",
        marker=dict(symbol="circle", color="red", size=7)
    ))
    fig.add_trace(go.Scatter(
        x=sample_idx[mask_good],
        y=pred_vals[mask_good],
        mode="markers",
        name="Good (Pred)",
        marker=dict(symbol="circle", color="blue", size=7)
    ))

    # BAD
    fig.add_trace(go.Scatter(
        x=sample_idx[mask_bad],
        y=true_vals[mask_bad],
        mode="markers",
        name="Bad (True)",
        marker=dict(symbol="x", color="red", size=9)
    ))
    fig.add_trace(go.Scatter(
        x=sample_idx[mask_bad],
        y=pred_vals[mask_bad],
        mode="markers",
        name="Bad (Pred)",
        marker=dict(symbol="x", color="blue", size=9)
    ))

    # Explode
    fig.add_trace(go.Scatter(
        x=sample_idx[mask_explode],
        y=true_vals[mask_explode],
        mode="markers",
        name="Explode (True)",
        marker=dict(symbol="triangle-up", color="red", size=9)
    ))
    fig.add_trace(go.Scatter(
        x=sample_idx[mask_explode],
        y=pred_vals[mask_explode],
        mode="markers",
        name="Explode (Pred)",
        marker=dict(symbol="triangle-up", color="blue", size=9)
    ))

    # Connecting lines
    for i in range(len(sample_idx)):
        fig.add_trace(go.Scatter(
            x=[sample_idx[i], sample_idx[i]],
            y=[true_vals[i], pred_vals[i]],
            mode="lines",
            line=dict(color="gray", width=1),
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Sample",
        yaxis_title="Pull Force",
        template="seaborn"
    )

    fig.show()



def plot_visualizer_classification(y_true, y_pred, class_labels, title="Confusion Matrix"):
    """
    Plot confusion matrix and print F1 score for classification.

    Parameters
    ----------
    y_true : array-like
        True category labels for the fold.
    y_pred : array-like
        Predicted category labels for the fold.
    class_labels : list
        List of class names in desired order, e.g. ["Bad", "Good"].
    title : str
        Title for the confusion matrix plot.
    """

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_labels)

    disp.plot(cmap="Blues", values_format='d')
    plt.title(title)
    plt.show()

    # F1 Score
    f1 = f1_score(y_true, y_pred, average="macro")
    print("F1 Score:", f1)




