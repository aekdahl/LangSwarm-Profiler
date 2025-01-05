# db/chroma_db.py
from typing import List, Optional
import json

from .base import DBBase

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError as e:
    raise ImportError(
        "ChromaDB is not installed.\n"
        "Please install it with: pip install chromadb\n"
        f"Error: {e}"
    )


class ChromaDB(DBBase):
    def __init__(self, persist_directory: str):
        self.client = chromadb.Client()
        self.collection = self._create_collection(persist_directory)

    def _create_collection(self, persist_directory: str):
        return self.client.create_collection(
            name="langprofiler",
            embedding_function=None,  # Define if using embedding functions
            persist_directory=persist_directory
        )

    def add_agent(self, agent_id: str, agent_info: str, features: str):
        # ChromaDB does not have tables, but you can store agent info as metadata
        metadata = {
            "agent_id": agent_id,
            "agent_info": agent_info,
            "features": features,
            "length_of_prompt": 0,
            "conciseness": 0.0
        }
        # Embedding can be a dummy vector or actual agent embedding
        dummy_vector = [0.0] * 768  # Example vector length
        self.collection.add(
            documents=["Agent Info"],
            embeddings=[dummy_vector],
            metadatas=[metadata],
            ids=[agent_id]
        )

    def get_agent(self, agent_id: str):
        results = self.collection.get(ids=[agent_id], include=["metadatas"])
        if results and results['metadatas']:
            return results['metadatas'][0]
        return None

    def add_interaction(self, interaction_data: dict):
        metadata = {
            "agent_id": interaction_data["agent_id"],
            "query": interaction_data["query"],
            "response": interaction_data["response"],
            "timestamp": interaction_data["timestamp"],
            "latency": interaction_data["latency"],
            "feedback": interaction_data["feedback"],
            "features": interaction_data["features"],
            "length_of_prompt": interaction_data.get("length_of_prompt", 0),
            "conciseness": interaction_data.get("conciseness", 0.0)
        }
        dummy_vector = [0.0] * 768  # Example vector length
        interaction_id = f"{interaction_data['agent_id']}_{interaction_data['timestamp']}"
        self.collection.add(
            documents=[interaction_data["query"]],
            embeddings=[dummy_vector],
            metadatas=[metadata],
            ids=[interaction_id]
        )

    def list_interactions(self, agent_id: str):
        # ChromaDB does not support direct querying by metadata fields.
        # Implement filtering in application logic after retrieving all interactions.
        results = self.collection.get(include=["metadatas"])
        interactions = [
            metadata for metadata in results['metadatas']
            if metadata.get("agent_id") == agent_id
        ]
        return interactions

    def update_profile(self, agent_id: str, profile_vec: List[float]):
        # Store profile vector as metadata or manage separately
        # Example: Update agent's metadata with profile vector
        agent_metadata = self.get_agent(agent_id)
        if agent_metadata:
            agent_metadata['profile_vec'] = json.dumps(profile_vec)
            self.collection.update(
                ids=[agent_id],
                metadatas=[agent_metadata]
            )

    def get_profile(self, agent_id: str):
        agent_metadata = self.get_agent(agent_id)
        if agent_metadata and 'profile_vec' in agent_metadata:
            return {"profile_vec": json.loads(agent_metadata['profile_vec'])}
        return None

    def add_prompt(self, prompt_id: str, prompt_info: str, features: str):
        metadata = {
            "prompt_id": prompt_id,
            "prompt_info": prompt_info,
            "features": features,
            "length_of_prompt": 0,
            "conciseness": 0.0
        }
        dummy_vector = [0.0] * 768  # Example vector length
        self.collection.add(
            documents=["Prompt Info"],
            embeddings=[dummy_vector],
            metadatas=[metadata],
            ids=[prompt_id]
        )

    def get_prompt(self, prompt_id: str):
        results = self.collection.get(ids=[prompt_id], include=["metadatas"])
        if results and results['metadatas']:
            return results['metadatas'][0]
        return None

    def add_prompt_interaction(self, interaction_data: dict):
        metadata = {
            "prompt_id": interaction_data["prompt_id"],
            "query": interaction_data["query"],
            "response": interaction_data["response"],
            "timestamp": interaction_data["timestamp"],
            "latency": interaction_data["latency"],
            "feedback": interaction_data["feedback"],
            "features": interaction_data["features"],
            "length_of_prompt": interaction_data.get("length_of_prompt", 0),
            "conciseness": interaction_data.get("conciseness", 0.0)
        }
        dummy_vector = [0.0] * 768  # Example vector length
        prompt_interaction_id = f"{interaction_data['prompt_id']}_{interaction_data['timestamp']}"
        self.collection.add(
            documents=[interaction_data["query"]],
            embeddings=[dummy_vector],
            metadatas=[metadata],
            ids=[prompt_interaction_id]
        )

    def list_prompt_interactions(self, prompt_id: str):
        results = self.collection.get(include=["metadatas"])
        interactions = [
            metadata for metadata in results['metadatas']
            if metadata.get("prompt_id") == prompt_id
        ]
        return interactions

    def update_prompt_profile(self, prompt_id: str, profile_vec: List[float]):
        # Store prompt profile vector as metadata or manage separately
        # Example: Update prompt's metadata with profile vector
        prompt_metadata = self.get_prompt(prompt_id)
        if prompt_metadata:
            prompt_metadata['profile_vec'] = json.dumps(profile_vec)
            self.collection.update(
                ids=[prompt_id],
                metadatas=[prompt_metadata]
            )

    def get_prompt_profile(self, prompt_id: str):
        prompt_metadata = self.get_prompt(prompt_id)
        if prompt_metadata and 'profile_vec' in prompt_metadata:
            return {"profile_vec": json.loads(prompt_metadata['profile_vec'])}
        return None
