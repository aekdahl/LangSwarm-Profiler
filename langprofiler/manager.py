# manager.py

import time
from typing import Optional, List, Dict
import torch
import json

from .db.base import DBBase
from .db.sqlite_db import SqliteDB
from .db.chroma_db import ChromaDB
from .db.custom_sql import CustomSQLConnector
from .aggregator import HybridAggregatorNN
from .config import ProfilerConfig
from .feature_extractor import FeatureExtractor


class LangProfiler:
    """
    High-level manager class that handles Agents and Prompts with dynamic feature extraction.
    """
    def __init__(self, config: Optional[ProfilerConfig] = None, device: str = "cpu"):
        self.config = config or ProfilerConfig()
        self.db = self._init_db()
        self.aggregator = self._init_aggregator()
        self.feature_extractor = FeatureExtractor(device=device)
        self.feature_order = self.config.FEATURE_ORDER  # List[str]

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
            additional_feature_dim=len(self.feature_order),  # Reflect the number of additional features
            final_dim=self.config.AGGREGATOR_FINAL_DIM,
            do_normalize=True
        )
        return aggregator

    # -----------------
    # AGENT METHODS
    # -----------------
    def register_agent(self, agent_id: str, agent_info: dict, features: List[str]):
        """
        Registers a new agent with specified features.

        :param agent_id: Unique identifier for the agent.
        :param agent_info: Dictionary containing agent information.
        :param features: List of feature types to extract (e.g., ["intent", "sentiment"]).
        """
        agent_info_json = json.dumps(agent_info)
        features_json = json.dumps(features)
        self.db.add_agent(agent_id, agent_info_json, features_json)

    def get_agent_info(self, agent_id: str) -> Optional[dict]:
        """
        Retrieves information for the specified agent.

        :param agent_id: Unique identifier for the agent.
        :return: Dictionary containing agent information and features, or None if not found.
        """
        agent = self.db.get_agent(agent_id)
        if agent:
            agent_info = json.loads(agent['agent_info'])
            features = json.loads(agent['features'])
            agent_info['features'] = features
            return agent_info
        return None

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
        Logs an interaction for the specified agent.

        :param agent_id: Unique identifier for the agent.
        :param query: The user's query.
        :param response: The agent's response.
        :param latency: Response latency.
        :param feedback: User feedback score.
        :param timestamp: Timestamp of the interaction.
        """
        if timestamp is None:
            timestamp = time.time()

        # Retrieve agent's features
        agent = self.db.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent '{agent_id}' not found.")
        feature_types = json.loads(agent['features'])

        # Extract specified features
        extracted_features = self.feature_extractor.extract_features(query, feature_types)

        # Store extracted features as JSON
        features_json = json.dumps(extracted_features)

        interaction_data = {
            "agent_id": agent_id,
            "query": query,
            "response": response,
            "timestamp": timestamp,
            "latency": latency,
            "feedback": feedback,
            "features": features_json
        }
        self.db.add_interaction(interaction_data)
        # Optionally, update profile in real-time:
        self.update_profile(agent_id)

    def update_profile(self, agent_id: str):
        """
        Updates the profile vector for the specified agent based on interactions.

        :param agent_id: Unique identifier for the agent.
        """
        agent = self.db.get_agent(agent_id)
        if not agent:
            return

        agent_info = json.loads(agent['agent_info'])
        instructions_text = agent_info.get("instructions", "")
        numeric_features = agent_info.get("numeric_features", {})
        feature_types = json.loads(agent['features'])

        # Retrieve interactions to extract features
        interactions = self.db.list_interactions(agent_id)
        features = [json.loads(ix['features']) for ix in interactions]

        # Aggregate features based on feature types
        aggregated_features = {}
        for feature in feature_types:
            feature_lower = feature.lower()
            if feature_lower in self.feature_order:
                feature_values = [ix.get(feature_lower) for ix in features if ix.get(feature_lower) is not None]
                if not feature_values:
                    aggregated_features[feature_lower] = 0.0  # Default value if no data
                    continue
                first_value = feature_values[0]
                if isinstance(first_value, (int, float)):
                    # Average numerical features
                    aggregated_features[feature_lower] = sum(feature_values) / len(feature_values)
                elif isinstance(first_value, str):
                    # Mode for categorical features
                    aggregated_features[feature_lower] = Counter(feature_values).most_common(1)[0][0]
                elif isinstance(first_value, bool):
                    # Majority vote for boolean features
                    aggregated_features[feature_lower] = int(sum(feature_values) / len(feature_values) > 0.5)
                else:
                    # For lists or complex structures, use counts or other aggregations
                    if isinstance(first_value, list):
                        aggregated_features[feature_lower] = len(first_value)
                    else:
                        aggregated_features[feature_lower] = 0.0
            else:
                aggregated_features[feature_lower] = 0.0  # Default encoding

        # Prepare additional features list based on FEATURE_ORDER from config
        additional_features = [aggregated_features.get(ft.lower(), 0.0) for ft in self.feature_order]

        with torch.no_grad():
            embedding_tensor = self.aggregator(
                instructions_text,
                list(numeric_features.values()),  # Convert dict to list based on predefined order
                additional_features
            )
            embedding_vec = embedding_tensor[0].tolist()

        self.db.update_profile(agent_id, embedding_vec)

    def encode_feature(self, feature: str, feature_type: str) -> float:
        """
        Encodes the feature into a numerical value based on feature_type.

        :param feature: The feature string (e.g., intent label).
        :param feature_type: The type of feature (e.g., 'intent', 'topic').
        :return: Encoded numerical value.
        """
        feature_type = feature_type.lower()
        if feature_type == "intent":
            intent_mapping = {
                "financial planning": 1.0,
                "weather inquiry": 2.0,
                "entertainment": 3.0,
                "technical guidance": 4.0,
                "general information": 5.0,
                "educational inquiry": 6.0,
                "unknown": 0.0
            }
            return intent_mapping.get(feature.lower(), 0.0)
        elif feature_type == "sentiment":
            sentiment_mapping = {
                "positive": 1.0,
                "negative": -1.0,
                "neutral": 0.0
            }
            return sentiment_mapping.get(feature.lower(), 0.0)
        elif feature_type == "topic":
            # Example encoding for topic
            topic_mapping = {
                "machine learning": 1.0,
                "weather": 2.0,
                "entertainment": 3.0,
                "finance": 4.0,
                "unknown": 0.0
            }
            return topic_mapping.get(feature.lower(), 0.0)
        elif feature_type == "readability_score":
            # Example encoding for readability_score
            return float(feature)  # Assuming it's a float already
        elif feature_type == "key_phrase_extraction":
            # Example encoding for key phrases
            # Could be the count of key phrases or another logic
            return float(len(feature)) if isinstance(feature, list) else 0.0
        elif feature_type == "temporal_features":
            # Example encoding for temporal features
            return float(len(feature)) if isinstance(feature, list) else 0.0
        elif feature_type == "length_of_prompt":
            return float(feature)
        elif feature_type == "conciseness":
            return float(feature)
        else:
            return 0.0  # Default encoding for unsupported features

    def get_current_profile(self, agent_id: str) -> Optional[List[float]]:
        """
        Retrieves the current profile vector for the specified agent.

        :param agent_id: Unique identifier for the agent.
        :return: Profile vector as a list of floats, or None if not found.
        """
        profile = self.db.get_profile(agent_id)
        if profile:
            return json.loads(profile['profile_vec'])
        return None

    def list_interactions(self, agent_id: str) -> List[dict]:
        """
        Lists all interactions for the specified agent.

        :param agent_id: Unique identifier for the agent.
        :return: List of interaction dictionaries.
        """
        interactions = self.db.list_interactions(agent_id)
        # Convert JSON strings to dictionaries
        return [
            {
                "interaction_id": ix['interaction_id'],
                "agent_id": ix['agent_id'],
                "query": ix['query'],
                "response": ix['response'],
                "timestamp": ix['timestamp'],
                "latency": ix['latency'],
                "feedback": ix['feedback'],
                "features": json.loads(ix['features'])
            }
            for ix in interactions
        ]

    # -----------------
    # PROMPT METHODS
    # -----------------
    def register_prompt(self, prompt_id: str, prompt_info: dict, features: List[str]):
        """
        Registers a new prompt with specified features.

        :param prompt_id: Unique identifier for the prompt.
        :param prompt_info: Dictionary containing prompt information.
        :param features: List of feature types to extract (e.g., ["topic", "sentiment"]).
        """
        prompt_info_json = json.dumps(prompt_info)
        features_json = json.dumps(features)
        self.db.add_prompt(prompt_id, prompt_info_json, features_json)

    def get_prompt_info(self, prompt_id: str) -> Optional[dict]:
        """
        Retrieves information for the specified prompt.

        :param prompt_id: Unique identifier for the prompt.
        :return: Dictionary containing prompt information and features, or None if not found.
        """
        prompt = self.db.get_prompt(prompt_id)
        if prompt:
            prompt_info = json.loads(prompt['prompt_info'])
            features = json.loads(prompt['features'])
            prompt_info['features'] = features
            return prompt_info
        return None

    def log_prompt_interaction(
        self,
        prompt_id: str,
        query: str,
        response: str,
        latency: float = 0.0,
        feedback: float = 0.0
    ):
        """
        Logs an interaction for the specified prompt.

        :param prompt_id: Unique identifier for the prompt.
        :param query: The user's query.
        :param response: The prompt's response.
        :param latency: Response latency.
        :param feedback: User feedback score.
        """
        timestamp = time.time()

        # Retrieve prompt's features
        prompt = self.db.get_prompt(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_id}' not found.")
        feature_types = json.loads(prompt['features'])

        # Extract specified features
        extracted_features = self.feature_extractor.extract_features(query, feature_types)

        # Store extracted features as JSON
        features_json = json.dumps(extracted_features)

        interaction_data = {
            "prompt_id": prompt_id,
            "query": query,
            "response": response,
            "timestamp": timestamp,
            "latency": latency,
            "feedback": feedback,
            "features": features_json
        }
        self.db.add_prompt_interaction(interaction_data)
        # Optionally, update profile in real-time:
        self.update_prompt_profile(prompt_id)

    def update_prompt_profile(self, prompt_id: str):
        """
        Updates the profile vector for the specified prompt based on interactions.

        :param prompt_id: Unique identifier for the prompt.
        """
        prompt = self.db.get_prompt(prompt_id)
        if not prompt:
            return

        prompt_info = json.loads(prompt['prompt_info'])
        instructions_text = prompt_info.get("text", "")
        numeric_features = prompt_info.get("numeric_features", {})
        feature_types = json.loads(prompt['features'])

        # Retrieve prompt interactions to extract features
        interactions = self.db.list_prompt_interactions(prompt_id)
        features = [json.loads(ix['features']) for ix in interactions]

        # Aggregate features based on feature types
        aggregated_features = {}
        for feature in feature_types:
            feature_lower = feature.lower()
            if feature_lower in self.feature_order:
                feature_values = [ix.get(feature_lower) for ix in features if ix.get(feature_lower) is not None]
                if not feature_values:
                    aggregated_features[feature_lower] = 0.0  # Default value if no data
                    continue
                first_value = feature_values[0]
                if isinstance(first_value, (int, float)):
                    # Average numerical features
                    aggregated_features[feature_lower] = sum(feature_values) / len(feature_values)
                elif isinstance(first_value, str):
                    # Mode for categorical features
                    aggregated_features[feature_lower] = Counter(feature_values).most_common(1)[0][0]
                elif isinstance(first_value, bool):
                    # Majority vote for boolean features
                    aggregated_features[feature_lower] = int(sum(feature_values) / len(feature_values) > 0.5)
                else:
                    # For lists or complex structures, use counts or other aggregations
                    if isinstance(first_value, list):
                        aggregated_features[feature_lower] = len(first_value)
                    else:
                        aggregated_features[feature_lower] = 0.0
            else:
                aggregated_features[feature_lower] = 0.0  # Default encoding

        # Prepare additional features list based on FEATURE_ORDER from config
        additional_features = [aggregated_features.get(ft.lower(), 0.0) for ft in self.feature_order]

        with torch.no_grad():
            embedding_tensor = self.aggregator(
                instructions_text,
                list(numeric_features.values()),  # Convert dict to list based on predefined order
                additional_features
            )
            embedding_vec = embedding_tensor[0].tolist()

        self.db.update_prompt_profile(prompt_id, embedding_vec)

    def get_prompt_profile(self, prompt_id: str) -> Optional[List[float]]:
        """
        Retrieves the current profile vector for the specified prompt.

        :param prompt_id: Unique identifier for the prompt.
        :return: Profile vector as a list of floats, or None if not found.
        """
        profile = self.db.get_prompt_profile(prompt_id)
        if profile:
            return json.loads(profile['profile_vec'])
        return None

    def list_prompt_interactions(self, prompt_id: str) -> List[dict]:
        """
        Lists all interactions for the specified prompt.

        :param prompt_id: Unique identifier for the prompt.
        :return: List of prompt interaction dictionaries.
        """
        interactions = self.db.list_prompt_interactions(prompt_id)
        # Convert JSON strings to dictionaries
        return [
            {
                "prompt_interaction_id": ix['prompt_interaction_id'],
                "prompt_id": ix['prompt_id'],
                "query": ix['query'],
                "response": ix['response'],
                "timestamp": ix['timestamp'],
                "latency": ix['latency'],
                "feedback": ix['feedback'],
                "features": json.loads(ix['features'])
            }
            for ix in interactions
        ]
