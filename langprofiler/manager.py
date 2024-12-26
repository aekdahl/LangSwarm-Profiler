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
            additional_feature_dim=self.config.AGGREGATOR_ADDITIONAL_FEATURE_DIM,
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
        numeric_features = agent_info.get("numeric_features", [])
        feature_types = json.loads(agent['features'])
        
        # Retrieve interactions to extract features
        interactions = self.db.list_interactions(agent_id)
        features = [json.loads(ix['features']) for ix in interactions]
        
        # Aggregate features based on feature types
        aggregated_features = {}
        for feature in feature_types:
            feature_lower = feature.lower()
            if feature_lower in ["intent", "topic", "sentiment"]:
                # Collect all values for the feature
                feature_values = [ix.get(feature_lower, "Unknown") for ix in features]
                if feature_lower == "sentiment":
                    # For sentiment, compute average sentiment score
                    sentiment_mapping = {
                        "POSITIVE": 1.0,
                        "NEGATIVE": -1.0,
                        "NEUTRAL": 0.0
                    }
                    numerical_sentiments = [sentiment_mapping.get(sent, 0.0) for sent in feature_values]
                    average_sentiment = sum(numerical_sentiments) / len(numerical_sentiments) if numerical_sentiments else 0.0
                    aggregated_features["sentiment"] = average_sentiment
                else:
                    # For categorical features like intent and topic, use the most common value
                    from collections import Counter
                    counter = Counter(feature_values)
                    most_common = counter.most_common(1)[0][0]
                    # Encode the categorical feature
                    aggregated_features[feature_lower] = self.encode_feature(most_common, feature_type=feature_lower)
            else:
                # Handle other feature types as needed
                aggregated_features[feature_lower] = 0.0  # Default encoding
        
        # Prepare additional features list
        # Ensure the order matches the aggregator's expected input
        additional_features = [aggregated_features.get(ft.lower(), 0.0) for ft in feature_types]
        
        with torch.no_grad():
            embedding_tensor = self.aggregator(
                instructions_text, 
                numeric_features,
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
                "Financial Planning": 1.0,
                "Weather Inquiry": 2.0,
                "Entertainment": 3.0,
                "Technical Guidance": 4.0,
                "General Information": 5.0,
                "Educational Inquiry": 6.0,
                "Unknown": 0.0
            }
            return intent_mapping.get(feature, 0.0)
        elif feature_type == "sentiment":
            sentiment_mapping = {
                "POSITIVE": 1.0,
                "NEGATIVE": -1.0,
                "NEUTRAL": 0.0
            }
            return sentiment_mapping.get(feature.upper(), 0.0)
        elif feature_type == "topic":
            # For simplicity, assign unique values to topics
            # In a real scenario, you might have a dynamic or more sophisticated mapping
            topic_mapping = {
                "Machine Learning": 1.0,
                "Weather": 2.0,
                "Entertainment": 3.0,
                "Financial Planning": 4.0,
                "Unknown": 0.0
            }
            return topic_mapping.get(feature, 0.0)
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
        numeric_features = prompt_info.get("numeric_features", [])
        feature_types = json.loads(prompt['features'])
        
        # Retrieve prompt interactions to extract features
        interactions = self.db.list_prompt_interactions(prompt_id)
        features = [json.loads(ix['features']) for ix in interactions]
        
        # Aggregate features based on feature types
        aggregated_features = {}
        for feature in feature_types:
            feature_lower = feature.lower()
            if feature_lower in ["intent", "topic", "sentiment"]:
                # Collect all values for the feature
                feature_values = [ix.get(feature_lower, "Unknown") for ix in features]
                if feature_lower == "sentiment":
                    # For sentiment, compute average sentiment score
                    sentiment_mapping = {
                        "POSITIVE": 1.0,
                        "NEGATIVE": -1.0,
                        "NEUTRAL": 0.0
                    }
                    numerical_sentiments = [sentiment_mapping.get(sent, 0.0) for sent in feature_values]
                    average_sentiment = sum(numerical_sentiments) / len(numerical_sentiments) if numerical_sentiments else 0.0
                    aggregated_features["sentiment"] = average_sentiment
                else:
                    # For categorical features like intent and topic, use the most common value
                    from collections import Counter
                    counter = Counter(feature_values)
                    most_common = counter.most_common(1)[0][0]
                    # Encode the categorical feature
                    aggregated_features[feature_lower] = self.encode_feature(most_common, feature_type=feature_lower)
            else:
                # Handle other feature types as needed
                aggregated_features[feature_lower] = 0.0  # Default encoding
        
        # Prepare additional features list
        # Ensure the order matches the aggregator's expected input
        additional_features = [aggregated_features.get(ft.lower(), 0.0) for ft in feature_types]
        
        with torch.no_grad():
            embedding_tensor = self.aggregator(
                instructions_text, 
                numeric_features,
                additional_features
            )
            embedding_vec = embedding_tensor[0].tolist()
        
        self.db.update_profile(prompt_id, embedding_vec)
