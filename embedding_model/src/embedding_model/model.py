import numpy as np
import pandas as pd
import torch
from torch.nn import (
    Module,
    Embedding,
    Linear,
    ReLU,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from gensim.models import KeyedVectors
from embedding_model.utils import get_metrics, train_w2v

class DTNN1(Module):
    def __init__(self, embedding_weights, num_bins):
        super(DTNN1, self).__init__()
        self.embedding = Embedding.from_pretrained(embedding_weights, freeze=True)
        self.parallel = Linear(embedding_weights.shape[1], 256)
        self.relu = ReLU()
        self.fc1 = Linear(in_features=256, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=num_bins+1)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x):
        # input x shape = ([batch_size, seq_length])
        #x = self.embedding(x.long()) # forced type casting because SHAP explainer requires float inputs
        x = self.embedding(x)# shape = ([batch_size, seq_length, emb_dim])
        x = torch.stack([self.relu(self.parallel(embedding)) for embedding in x]) # shape = ([batch_size, seq_length, hidden_dim])
        x = torch.mean(x, dim=1)  # shape = ([batch_size, hidden_dim])
        x = self.fc1(x)  # shape = ([batch_size, hidden_dim])
        x = self.relu(x)  # shape = ([batch_size, hidden_dim])
        x = self.softmax(self.fc2(x))  # shape = ([batch_size, num_bins+1])

        return x

class DTNN2(Module):
    def __init__(self, embedding_weights, num_bins, n_layers, n_heads, dropout):
        super(DTNN2, self).__init__()
        self.embedding = Embedding.from_pretrained(embedding_weights, freeze=True)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=256, nhead=n_heads, dim_feedforward=256, dropout=dropout, batch_first=True
            ),
            num_layers=n_layers,
        )
        self.relu = ReLU()
        self.fc1 = Linear(in_features=256, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=num_bins+1)
        self.softmax = torch.nn.Softmax(-1)
        

    def forward(self, x):
        # input x shape = ([batch_size, seq_length])
        x = self.embedding(x)  # shape = ([batch_size, seq_length, emb_dim])
        x = self.transformer_encoder(x) # shape = ([batch_size, seq_length, hidden_dim])
        x = torch.mean(x, dim=1)  # shape = ([batch_size, hidden_dim])
        x = self.fc1(x)  # shape = ([batch_size, hidden_dim])
        x = self.relu(x)  # shape = ([batch_size, hidden_dim])
        x = self.softmax(self.fc2(x))  # shape = ([batch_size, num_bins+1])

        return x

class DCPH1(Module):
    def __init__(self, embedding_weights):
        super(DCPH1, self).__init__()
        self.embedding = Embedding.from_pretrained(embedding_weights, freeze=True)
        self.parallel = Linear(embedding_weights.shape[1], 256)
        self.relu = ReLU()
        self.fc1 = Linear(in_features=256, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=1)

    def forward(self, x):
        # input x shape = ([batch_size, seq_length])
        x = self.embedding(x)  # shape = ([batch_size, seq_length, emb_dim])
        x = torch.stack([self.relu(self.parallel(embedding)) for embedding in x]) # shape = ([batch_size, seq_length, hidden_dim])
        x = torch.mean(x, dim=1)  # shape = ([batch_size, hidden_dim])
        x = self.fc1(x)  # shape = ([batch_size, hidden_dim])
        x = self.relu(x)  # shape = ([batch_size, hidden_dim])
        x = self.fc2(x)  # shape = ([batch_size, output_dim])

        return x

    def compute_baseline_hazards(self, log_hazards, labels, durations):
        df = pd.DataFrame({
            'log_hazards': log_hazards,
            'labels': labels,
            'durations': durations
        })

        baseline_hazards = (
            df
            .assign(expg=np.exp(log_hazards))
            .groupby('durations')
            .agg({'expg': 'sum', 'labels': 'sum'})
            .sort_index(ascending=False)
            .assign(expg=lambda x: x['expg'].cumsum())
            .pipe(lambda x: x['labels']/x['expg'])
            .fillna(0.)
            .iloc[::-1]
            .loc[lambda x: x.index <= df['durations'].max()]
            .rename('baseline_hazards'))

        return baseline_hazards

    def predict_cumulative_hazards(self, baseline_hazards, log_hazards):
        bch = baseline_hazards.cumsum().rename('baseline_cumulative_hazards')
        expg = np.exp(log_hazards).reshape(1,-1)
        return pd.DataFrame(bch.values.reshape(-1,1).dot(expg), index=bch.index)

    def predict_surv(self, baseline_hazards):
        return np.exp(-baseline_hazards.cumsum())

class BC1(Module):
    def __init__(self, embedding_weights):
        super(BC1, self).__init__()
        self.embedding = Embedding.from_pretrained(embedding_weights, freeze=True)
        self.fc = Linear(in_features=embedding_weights.shape[1], out_features=1)

    def forward(self, x):
        # input x shape = ([batch_size, seq_length])
        x = self.embedding(x)  # shape = ([batch_size, seq_length, emb_dim])
        x = torch.mean(x, dim=1)  # shape = ([batch_size, emb_dim])
        x = self.fc(x)  # shape = ([batch_size, output_dim])

        return x


class BC2(Module):
    def __init__(self, embedding_weights):
        super(BC2, self).__init__()
        self.embedding = Embedding.from_pretrained(embedding_weights, freeze=True)
        self.relu = ReLU()
        self.fc1 = Linear(in_features=embedding_weights.shape[1], out_features=256)
        self.fc2 = Linear(in_features=256, out_features=1)

    def forward(self, x):
        # input x shape = ([batch_size, seq_length])
        x = self.embedding(x)  # shape = ([batch_size, seq_length, emb_dim])
        x = torch.mean(x, dim=1)  # shape = ([batch_size, emb_dim])
        x = self.fc1(x)  # shape = ([batch_size, hidden_dim])
        x = self.relu(x)  # shape = ([batch_size, hidden_dim])
        x = self.fc2(x)  # shape = ([batch_size, output_dim])

        return x

class BC3(Module):
    def __init__(self, embedding_weights):
        super(BC3, self).__init__()
        self.embedding = Embedding.from_pretrained(embedding_weights, freeze=True)
        self.parallel = Linear(embedding_weights.shape[1], 256)
        self.relu = ReLU()
        self.fc1 = Linear(in_features=256, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=1)

    def forward(self, x):
        # input x shape = ([batch_size, seq_length])
        x = self.embedding(x)  # shape = ([batch_size, seq_length, emb_dim])
        x = torch.stack([self.relu(self.parallel(embedding)) for embedding in x]) # shape = ([batch_size, seq_length, hidden_dim])
        x = torch.mean(x, dim=1)  # shape = ([batch_size, hidden_dim])
        x = self.fc1(x)  # shape = ([batch_size, hidden_dim])
        x = self.relu(x)  # shape = ([batch_size, hidden_dim])
        x = self.fc2(x)  # shape = ([batch_size, output_dim])

        return x


class BC4(Module):
    def __init__(self, embedding_weights, n_layers, n_heads, dropout):
        super(BC4, self).__init__()
        self.embedding = Embedding.from_pretrained(embedding_weights, freeze=True)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=256, nhead=n_heads, dim_feedforward=256, dropout=dropout, batch_first=True
            ),
            num_layers=n_layers,
        )
        self.relu = ReLU()
        self.fc1 = Linear(in_features=256, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=1)

    def forward(self, x):
        # input x shape = ([batch_size, seq_length])
        x = self.embedding(x)  # shape = ([batch_size, seq_length, emb_dim])
        x = self.transformer_encoder(x)  # shape = ([batch_size, seq_length, emb_dim])
        x = torch.mean(x, dim=1)  # shape = ([batch_size, emb_dim])
        x = self.fc1(x)  # shape = ([batch_size, hidden_dim])
        x = self.relu(x)  # shape = ([batch_size, hidden_dim])
        x = self.fc2(x)  # shape = ([batch_size, output_dim])

        return x

def run_epoch(model, device, data_loader, optimizer, criterion, model_type, training=True):
    """
    Runs one epoch (training or evaluation) on the provided data_loader
    Returns loss and metrics
    """
    model.train() if training else model.eval()
    loss_epoch = 0.0

    with torch.set_grad_enabled(training):
        for events, labels, event_times in data_loader:
            events, labels, event_times = events.to(device), labels.to(device), event_times.to(device)
            optimizer.zero_grad()
            predictions = model(events)

            if model_type.startswith("BC"):
                loss = criterion(predictions, labels)
            elif model_type.startswith("DTNN"):
                loss = criterion(predictions, labels.squeeze(dim=1), event_times.squeeze(dim=1))
            elif model_type.startswith("DCPH"):
                loss = criterion(predictions.squeeze(dim=1), event_times.squeeze(dim=1), labels.squeeze(dim=1))
            if training:
                loss.backward()
                optimizer.step()
            
            loss_epoch += loss.item()

    loss_epoch /= len(data_loader)

    return loss_epoch


def test_model(model, device, test_loader, bin_boundaries):
    test_labels = []
    test_predictions = []
    test_event_times = []
    model.eval()
    for events, labels, event_times in test_loader:
        events, labels, event_times = events.to(device), labels.to(device), event_times.to(device)
        predictions = model(events)

        test_labels.append(labels.squeeze().detach().cpu())
        test_predictions.append(predictions.squeeze().detach().cpu())
        test_event_times.append(event_times.squeeze().detach().cpu())

    test_metrics = get_metrics(
        np.concatenate(test_labels),
        np.concatenate(test_event_times),
        np.concatenate(test_predictions),
        bin_boundaries[1:],
    )

    return test_metrics

def init_model(model_name, config, label):
    """
    Initializes Embedding Model with word2vec embedding weights
    """
    phenotype = getattr(config.phenotypes, label)

    # Get w2v embeddings
    try:
        w2v = KeyedVectors.load(str(phenotype.filepaths.word2vec))
        embedding_weights = torch.tensor(w2v.vectors)
    except FileNotFoundError:
        print("word2vec not found, training embeddings now...")
        df = pd.read_parquet(phenotype.filepaths.train)
        sentences = df.groupby("mrn")["event"].agg(list).to_list()
        w2v_model = train_w2v(sentences)
        w2v_model.wv.save(str(phenotype.filepaths.word2vec))
        embedding_weights = torch.tensor(w2v_model.wv.vectors)
    print("Loaded word2vec embeddings")
    
    if model_name == "BC1":
        model = BC1(embedding_weights)
    elif model_name == "BC2":
        model = BC2(embedding_weights)
    elif model_name == "BC3":
        model = BC3(embedding_weights)
    elif model_name == "BC4":
        model = BC4(
            embedding_weights=embedding_weights,
            n_layers=config.model.num_layers,
            n_heads=config.model.num_heads,
            dropout=config.model.dropout
        )
    elif model_name == "DTNN1":
        model = DTNN1(
            embedding_weights=embedding_weights,
            num_bins=len(phenotype.bin_boundaries)-1
        )
    elif model_name == "DTNN2":
        model = DTNN2(
            embedding_weights=embedding_weights,
            num_bins=len(phenotype.bin_boundaries)-1,
            n_layers=config.model.num_layers,
            n_heads=config.model.num_heads,
            dropout=config.model.dropout
        )
    elif model_name == "DCPH1":
        model = DCPH1(embedding_weights)
    return model