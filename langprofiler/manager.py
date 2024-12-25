"""
Main interface to the LangProfiler functionality.
"""
# import numpy as np
# from .models import generate_agent_id, Agent, Interaction

# manager.py
import time
from typing import Optional, List

from .db.base import DBBase
from .db.sqlite_db import SqliteDB
from .db.chroma_db import ChromaDB
from .db.custom_sql import CustomSQLConnector
from .aggregator import HybridAggregatorNN
from .config import ProfilerConfig
import torch

class LangProfiler:
    """
    High-level manager class that ties together the DB layer and the aggregator.
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self.db = self._init_db()
        self.aggregator = self._init_aggregator()

    def _init_db(self) -> DBBase:
        backend = self.config.DB_BACKEND
        if backend == "sqlite":
            return SqliteDB(db_path=self.config.DB_DSN)
        elif backend == "chroma":
            return ChromaDB(persist_directory=self.config.CHROMA_PERSIST_DIR)
        elif backend == "custom_sql":
            return CustomSQLConnector(
                dsn=self.config.DB_DSN,
                init_ddl=self.config.CUSTOM_SQL_INIT_DDL
            )
        else:
            raise ValueError(f"Unknown DB backend: {backend}")

    def _init_aggregator(self) -> HybridAggregatorNN:
        """
        Creates a Sentence-BERT aggregator or another aggregator from aggregator.py
        """
        aggregator = HybridAggregatorNN(
            model_name=self.config.AGGREGATOR_MODEL_NAME,
            numeric_dim=self.config.AGGREGATOR_NUMERIC_DIM,
            final_dim=self.config.AGGREGATOR_FINAL_DIM
        )
        return aggregator

    # -----------------
    # Public API Methods
    # -----------------

    def register_agent(self, agent_id: str, agent_info: dict):
        """
        Register a new agent in the DB. Overwrite if agent_id already exists.
        agent_info might include 'name', 'cost', 'domain_tags', 'instructions', etc.
        """
        self.db.add_agent(agent_id, agent_info)

    def get_agent_info(self, agent_id: str) -> Optional[dict]:
        return self.db.get_agent(agent_id)

    def log_interaction(
        self,
        agent_id: str,
        query: str,
        response: str,
        latency: float = 0.0,
        feedback: float = 0.0,
        timestamp: float = None
    ):
        """
        Log a single interaction in the DB.
        Optionally call update_profile here or do it asynchronously.
        """
        if timestamp is None:
            timestamp = time.time()
        interaction_data = {
            "agent_id": agent_id,
            "query": query,
            "response": response,
            "timestamp": timestamp,
            "latency": latency,
            "feedback": feedback
        }
        self.db.add_interaction(interaction_data)
        # Optionally, update profile in real-time:
        self.update_profile(agent_id)

    def update_profile(self, agent_id: str):
        """
        Recomputes or updates the agent's profile embedding in the DB.
        Example: gather agent instructions + numeric features, pass to aggregator, store in DB.
        """
        agent_info = self.db.get_agent(agent_id)
        if not agent_info:
            # Can't update if we don't know the agent
            return

        # Example of instructions text + numeric features
        instructions_text = agent_info.get("instructions", "")
        # Suppose we store cost, domain flags, etc. as numeric
        numeric_features = agent_info.get("numeric_features", [])

        # aggregator(...) returns a batch, but let's handle single agent
        with torch.no_grad():
            embedding_tensor = self.aggregator(instructions_text, numeric_features)
            embedding_list = embedding_tensor[0].tolist()  # shape (final_dim,)

        self.db.update_profile(agent_id, embedding_list)

    def get_current_profile(self, agent_id: str) -> Optional[List[float]]:
        return self.db.get_profile(agent_id)

    def list_interactions(self, agent_id: str) -> List[dict]:
        return self.db.list_interactions(agent_id)

