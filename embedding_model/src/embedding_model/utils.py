import matplotlib.pyplot as plt
import torch
import numpy as np
from gensim.models import Word2Vec
import seaborn as sns
from multiprocessing import cpu_count
from sklearn.metrics import roc_auc_score, average_precision_score
from embedding_model.metrics import xAUCt, xAPt


def train_w2v(sentences):
    """
    Pretrains EHR embeddings using w2v model
    """
    w2v = Word2Vec(
        min_count=0, 
        window=3,
        vector_size=256,
        sample=6e-5, 
        alpha=0.01, 
        min_alpha=0.0001, 
        negative=20,
        workers=cpu_count() - 1)
    w2v.build_vocab(sentences)
    w2v.train(sentences, total_examples=w2v.corpus_count, epochs=30)

    # Augment the vocabulary with OOV and PAD vectors
    w2v.wv.add_vectors(['OOV', 'PAD'], [np.random.rand(256), np.zeros(256)])

    return w2v


def get_metrics(s, t, predictions, times):
    """
    Returns AUC and AP for each time bin
    """
    if predictions.ndim == 1:
        cumulative_predicted_risk = np.tile(predictions[:, np.newaxis], (1, len(times)))
    elif predictions.ndim == 2:
        cumulative_predicted_risk = np.cumsum(predictions, axis=1)[:, :-1]
    
    auct = xAUCt(s, t, cumulative_predicted_risk, times)
    apt = xAPt(s, t, cumulative_predicted_risk, times)

    metrics = {
        "AUC": roc_auc_score(s, cumulative_predicted_risk[:,-1]),
        "AP": average_precision_score(s, cumulative_predicted_risk[:,-1]),
        "AUCt": auct,
        "APt": apt,
    }

    return metrics

def plot_confusion_matrix(cm, label):
    """
    Plots seaborn heatmap for respective label
    """
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix for {label}")
    plt.show()


def pad_sequences(sequences, seq_threshold, padding_index):
    """
    Returns sequence of fixed length based on threshold
    - If given sequence is longer than threshold, truncate and take the n most recent sequences
    - If given sequence is shorter than threshold, pad the remaining sequence
    """
    seq_length = sequences.shape[0]
    if seq_length >= seq_threshold:
        # select the n most recent sequences
        sequences = sequences[-seq_threshold:]  # shape = ([seq_threshold])

    elif seq_length < seq_threshold:
        pad_length = seq_threshold - seq_length
        pad_tensor = torch.full((pad_length,), padding_index)
        sequences = torch.cat((pad_tensor, sequences), dim=0)  # shape = ([seq_length + pad_length]) = ([seq_threshold])
    return sequences


def get_inputs(df, label, vocab, seq_threshold):
    age_to_first_diagnosis = df[label + "_diagnosis_age"].iloc[0]
    df = df[df["age"] < age_to_first_diagnosis]
    
    if len(df) == 0:
        events = torch.full((seq_threshold,), vocab["PAD"])
    else:
        events = torch.tensor([vocab.get(event, vocab["OOV"]) for event in df["event"].to_list()])
        events = pad_sequences(events, seq_threshold, vocab["PAD"])

    return events


def get_logits(model, events):
    """
    Returns log odds of the model prediction
    """
    model.eval()
    with torch.no_grad():
        prediction = model(events)

    return prediction