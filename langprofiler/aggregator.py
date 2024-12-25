"""
Simple aggregator/encoder for converting metrics into an embedding.
"""

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sentence_transformers import SentenceTransformer
except ImportError:
    PCA = None
    SentenceTransformer = None

try:
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
except ImportError:
    nn = None
    SentenceTransformer = None

if nn is None and SentenceTransformer is None and PCA is None:
    raise ImportError(
        "Either scikit-learn or PyTorch and sentence-transformers must be installed.\n"
        "Please install them using one of the following commands:\n\n"
        "  pip install torch sentence-transformers\n\n"
        "Or, if you rather use PCA:\n"
        "  pip install scikit-learn sentence-transformers\n\n"
    )


class SimpleAggregator:
    def __init__(self, embedding_size=8):
        """
        Initialize a basic aggregator that produces an N-dimensional embedding.
        """
        self.embedding_size = embedding_size

    def aggregate(self, agent_data):
        """
        agent_data: dictionary containing aggregated metrics about an agent.
                    Example: {"avg_latency": 0.123, "avg_cost": 0.001, "avg_feedback": 4.5}

        returns: A numpy array of shape (embedding_size,)
        """
        # For a super-simple approach, just transform metrics into a fixed-size vector.
        # We'll "fake" it here by hashing or normalizing values. Expand as needed.

        # Example: Convert numeric values to a vector, then pad/truncate to embedding_size.
        metrics_vec = []

        for key, value in sorted(agent_data.items()):
            # Just a placeholder approach: convert the float to a scaled number
            metrics_vec.append(float(value))

        # Convert list to numpy array, pad or trim to embedding_size
        arr = np.array(metrics_vec, dtype=np.float32)
        if len(arr) < self.embedding_size:
            padding = np.zeros(self.embedding_size - len(arr), dtype=np.float32)
            arr = np.concatenate([arr, padding])
        else:
            arr = arr[:self.embedding_size]

        return arr


class HybridAggregatorNN(nn.Module):
    """
    A hybrid aggregator that:
      1) Encodes text with Sentence-BERT,
      2) Optionally incorporates numeric features,
      3) Projects everything down to a smaller final embedding via an NN.
    """

    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        numeric_dim: int = 0,
        final_dim: int = 32,
        do_normalize: bool = True
    ):
        """
        :param model_name: Name of the Sentence-BERT model to load.
        :param numeric_dim: How many numeric features we expect to concatenate.
        :param final_dim: Size of the output embedding.
        """
        super().__init__()
        # 1. Sentence-BERT model for text
        self.sbert_model = SentenceTransformer(model_name)
        self.sbert_dim = self.sbert_model.get_sentence_embedding_dimension()

        # 2. Feed-forward layers to reduce dimension
        combined_input_dim = self.sbert_dim + numeric_dim
        hidden_dim = max(final_dim * 2, 64)  # or whatever heuristic you like

        self.reducer = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, final_dim)
        )

        # 3. Whether to apply L2 normalization
        self.do_normalize = do_normalize

    def forward(self, texts, numeric_features=None):
        """
        :param texts: A single string or a list of strings (instructions).
        :param numeric_features: 
            - None (if no numeric features), or 
            - a list of float lists, shape [batch_size, numeric_dim].
        :return: A (batch_size x final_dim) tensor.
        """
        # 1. Encode text via Sentence-BERT
        # If it's a single string, wrap in a list for consistency
        if isinstance(texts, str):
            texts = [texts]
        text_embeddings = self.sbert_model.encode(
            texts, 
            convert_to_tensor=True  # returns a PyTorch tensor
        )  # shape: (batch_size, sbert_dim)

        batch_size = text_embeddings.size(0)
        
        # 2. Prepare numeric features
        if numeric_features is None:
            # If no numeric features, just pass a zero tensor
            numeric_dim = 0
            numeric_tensor = torch.zeros(batch_size, 0)
        else:
            numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32)
            if len(numeric_tensor.shape) == 1:
                # if user passed a single row, shape (numeric_dim,)
                numeric_tensor = numeric_tensor.unsqueeze(0)  # shape: (1, numeric_dim)

        # 3. Concatenate text embedding + numeric
        combined = torch.cat([text_embeddings, numeric_tensor], dim=1)

        # 4. Pass through NN to get final embedding
        final_embedding = self.reducer(combined)

        # 5. Optional L2 normalization
        if self.do_normalize:
            # L2 norm each row
            final_embedding = F.normalize(final_embedding, p=2, dim=1)
            
        return final_embedding


class SentenceBERTWithPCA:
    """
    Uses Sentence-BERT to get embeddings, then applies PCA for dimension reduction.
    Optionally merges numeric features after PCA or before PCA.
    """

    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        pca_dim: int = 32,
        combine_numeric: bool = False
    ):
        """
        :param model_name: Name of the Sentence-BERT model to load.
        :param pca_dim: Target dimension after PCA.
        :param combine_numeric: If True, merges numeric features into the embeddings 
                                before PCA (requires fitting PCA to combined data).
                                Otherwise, you can merge after PCA or skip numeric features.
        """
        self.sbert_model = SentenceTransformer(model_name)
        self.sbert_dim = self.sbert_model.get_sentence_embedding_dimension()
        self.pca_dim = pca_dim
        self.combine_numeric = combine_numeric

        # Initialize PCA object (not yet fitted)
        self.pca = PCA(n_components=self.pca_dim)
        self.is_fitted = False

    def fit(self, text_samples, numeric_features=None):
        """
        Fit PCA on a sample dataset. 
        This should be done on a representative set of instructions (and numeric features if combine_numeric=True).
        """
        # Encode text via SBERT
        if isinstance(text_samples, str):
            text_samples = [text_samples]

        text_embeddings = self.sbert_model.encode(text_samples, convert_to_numpy=True)

        if self.combine_numeric and numeric_features is not None:
            numeric_array = np.array(numeric_features, dtype=np.float32)
            if len(numeric_array.shape) == 1:
                numeric_array = numeric_array.reshape(1, -1)
            combined_embeddings = np.concatenate([text_embeddings, numeric_array], axis=1)
        else:
            combined_embeddings = text_embeddings

        self.pca.fit(combined_embeddings)
        self.is_fitted = True

    def transform(self, text_samples, numeric_features=None):
        """
        Transforms new data to PCA-reduced embeddings.
        """
        if not self.is_fitted:
            raise ValueError("PCA is not yet fitted. Call `fit` first with sample data.")
        
        if isinstance(text_samples, str):
            text_samples = [text_samples]

        text_embeddings = self.sbert_model.encode(text_samples, convert_to_numpy=True)

        # Merge numeric if combine_numeric=True
        if self.combine_numeric and numeric_features is not None:
            numeric_array = np.array(numeric_features, dtype=np.float32)
            if len(numeric_array.shape) == 1:
                numeric_array = numeric_array.reshape(1, -1)
            combined_embeddings = np.concatenate([text_embeddings, numeric_array], axis=1)
        else:
            combined_embeddings = text_embeddings

        reduced_embeddings = self.pca.transform(combined_embeddings)
        return reduced_embeddings
