import numpy as np
from langswarm.profiler.aggregator import SimpleAggregator, HybridAggregatorNN

def test_simple_aggregator():
    aggregator = SimpleAggregator(embedding_size=5)
    data = {"avg_latency": 0.1, "avg_cost": 0.02, "avg_feedback": 4.5}
    result = aggregator.aggregate(data)
    assert len(result) == 5
    assert np.isclose(result[0], 0.1)

def test_hybrid_aggregator_initialization():
    aggregator = HybridAggregatorNN(
        model_name="all-MiniLM-L6-v2",
        numeric_dim=3,
        final_dim=16
    )
    assert aggregator.sbert_model is not None
    assert aggregator.reducer is not None
