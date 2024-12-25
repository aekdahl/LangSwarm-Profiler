# manager.py (extended with prompts)
import time
from typing import Optional, List

import torch

from .db.base import DBBase
from .db.sqlite_db import SqliteDB
from .db.chroma_db import ChromaDB
from .db.custom_sql import CustomSQLConnector
from .aggregator import HybridAggregatorNN
from .config import ProfilerConfig

class LangProfiler:
    """
    High-level manager class that now handles Agents AND Prompts.
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
        aggregator = HybridAggregatorNN(
            model_name=self.config.AGGREGATOR_MODEL_NAME,
            numeric_dim=self.config.AGGREGATOR_NUMERIC_DIM,
            final_dim=self.config.AGGREGATOR_FINAL_DIM
        )
        return aggregator

    # -------------------------------------------------
    # AGENT METHODS (same as before)
    # -------------------------------------------------
    def register_agent(self, agent_id: str, agent_info: dict):
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
        # Optionally update profile in real-time
        self.update_profile(agent_id)

    def update_profile(self, agent_id: str):
        agent_info = self.db.get_agent(agent_id)
        if not agent_info:
            return

        instructions_text = agent_info.get("instructions", "")
        numeric_features = agent_info.get("numeric_features", [])

        with torch.no_grad():
            embedding_tensor = self.aggregator(instructions_text, numeric_features)
            embedding_vec = embedding_tensor[0].tolist()

        self.db.update_profile(agent_id, embedding_vec)

    def get_current_profile(self, agent_id: str) -> Optional[List[float]]:
        return self.db.get_profile(agent_id)

    def list_interactions(self, agent_id: str) -> List[dict]:
        return self.db.list_interactions(agent_id)

    # -------------------------------------------------
    # NEW PROMPT METHODS
    # -------------------------------------------------

    def register_prompt(self, prompt_id: str, prompt_info: dict):
        """
        Register a new prompt in the DB. Overwrites if prompt_id already exists.
        prompt_info might include 'text', 'domain_tags', 'numeric_features', etc.
        """
        self.db.add_prompt(prompt_id, prompt_info)

    def get_prompt_info(self, prompt_id: str) -> Optional[dict]:
        return self.db.get_prompt(prompt_id)

    def log_prompt_interaction(
        self,
        prompt_id: str,
        query: str,
        response: str,
        latency: float = 0.0,
        feedback: float = 0.0,
        timestamp: float = None
    ):
        if timestamp is None:
            timestamp = time.time()
        data = {
            "prompt_id": prompt_id,
            "query": query,
            "response": response,
            "timestamp": timestamp,
            "latency": latency,
            "feedback": feedback
        }
        self.db.add_prompt_interaction(data)
        # Optionally update prompt profile now:
        self.update_prompt_profile(prompt_id)

    def update_prompt_profile(self, prompt_id: str):
        """
        Recompute or update the prompt's profile vector.
        For example, we can encode 'text' + optional numeric features from prompt_info.
        """
        prompt_info = self.db.get_prompt(prompt_id)
        if not prompt_info:
            return

        prompt_text = prompt_info.get("text", "")
        numeric_features = prompt_info.get("numeric_features", [])

        with torch.no_grad():
            embedding_tensor = self.aggregator(prompt_text, numeric_features)
            embedding_vec = embedding_tensor[0].tolist()

        self.db.update_prompt_profile(prompt_id, embedding_vec)

    def get_prompt_profile(self, prompt_id: str) -> Optional[List[float]]:
        return self.db.get_prompt_profile(prompt_id)

    def list_prompt_interactions(self, prompt_id: str) -> List[dict]:
        return self.db.list_prompt_interactions(prompt_id)
