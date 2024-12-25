"""
Simple aggregator/encoder for converting metrics into an embedding.
"""

import numpy as np

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
