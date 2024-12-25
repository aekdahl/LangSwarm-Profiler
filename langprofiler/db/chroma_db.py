# db/chroma_db.py
from typing import List, Optional
import json

from .base import DBBase

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(
        "ChromaDB is not installed.\n"
        "Please install it with: pip install chromadb\n"
        f"Error: {e}"
    )

class ChromaDB(DBBase):
    """
    Minimal ChromaDB implementation handling Agents and Prompts.
    We store embeddings in separate collections:
      - 'agent_profiles' for agent embeddings
      - 'prompt_profiles' for prompt embeddings

    For metadata like agent_info, prompt_info, and interactions, 
    we keep them in local dicts or lists. (You can expand this as needed.)
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        :param persist_directory: Where Chroma will store data if persistence is enabled.
        """
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))

        # For agent embeddings
        self.agent_collection = self.client.create_collection("agent_profiles")

        # For prompt embeddings
        self.prompt_collection = self.client.create_collection("prompt_profiles")

        # In-memory storage for agent info & interactions
        self.agents = {}
        self.interactions = []

        # In-memory storage for prompts
        self.prompts = {}
        self.prompt_interactions = []

    # ========= AGENTS =========
    def add_agent(self, agent_id: str, agent_info: dict) -> None:
        self.agents[agent_id] = agent_info

    def get_agent(self, agent_id: str) -> Optional[dict]:
        return self.agents.get(agent_id)

    def add_interaction(self, interaction_data: dict) -> None:
        # Just store in memory
        self.interactions.append(interaction_data)

    def list_interactions(self, agent_id: str) -> List[dict]:
        return [
            i for i in self.interactions
            if i.get("agent_id") == agent_id
        ]

    def update_profile(self, agent_id: str, profile_vec: List[float]) -> None:
        # Delete old doc if it exists
        self.agent_collection.delete(ids=[agent_id])
        self.agent_collection.add(
            documents=["Profile for agent " + agent_id],
            metadatas=[{"agent_id": agent_id}],
            embeddings=[profile_vec],
            ids=[agent_id]
        )

    def get_profile(self, agent_id: str) -> Optional[List[float]]:
        result = self.agent_collection.get(ids=[agent_id])
        if result and len(result["embeddings"]) > 0:
            return result["embeddings"][0]
        return None

    # ========= PROMPTS =========
    def add_prompt(self, prompt_id: str, prompt_info: dict) -> None:
        self.prompts[prompt_id] = prompt_info

    def get_prompt(self, prompt_id: str) -> Optional[dict]:
        return self.prompts.get(prompt_id)

    def add_prompt_interaction(self, interaction_data: dict) -> None:
        self.prompt_interactions.append(interaction_data)

    def list_prompt_interactions(self, prompt_id: str) -> List[dict]:
        return [
            p for p in self.prompt_interactions
            if p.get("prompt_id") == prompt_id
        ]

    def update_prompt_profile(self, prompt_id: str, profile_vec: List[float]) -> None:
        # Remove old doc if it exists
        self.prompt_collection.delete(ids=[prompt_id])
        self.prompt_collection.add(
            documents=["Profile for prompt " + prompt_id],
            metadatas=[{"prompt_id": prompt_id}],
            embeddings=[profile_vec],
            ids=[prompt_id]
        )

    def get_prompt_profile(self, prompt_id: str) -> Optional[List[float]]:
        result = self.prompt_collection.get(ids=[prompt_id])
        if result and len(result["embeddings"]) > 0:
            return result["embeddings"][0]
        return None

class ChromaDB(DBBase):
    """
    A minimal ChromaDB implementation storing agent profiles as vectors.
    We'll store agent metadata and interactions in memory (dict), or you can
    adapt it to store them in a separate collection or a standard DB.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        :param persist_directory: Where Chroma will store data if persistence is enabled.
        """
        # Initialize Chroma client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        # Create a collection for agent profiles
        self.profiles_collection = self.client.create_collection("agent_profiles")

        # For minimal example, we'll store agent metadata & interactions in a dict
        # In a real system, you might create multiple collections or use a separate DB
        self.agents_memory = {}
        self.interactions_memory = []

    def add_agent(self, agent_id: str, agent_info: dict) -> None:
        self.agents_memory[agent_id] = agent_info

    def get_agent(self, agent_id: str) -> Optional[dict]:
        return self.agents_memory.get(agent_id)

    def add_interaction(self, interaction_data: dict) -> None:
        self.interactions_memory.append(interaction_data)

    def list_interactions(self, agent_id: str) -> List[dict]:
        return [ix for ix in self.interactions_memory if ix.get("agent_id") == agent_id]

    def update_profile(self, agent_id: str, profile_vec: List[float]) -> None:
        """
        Store or update the agent's profile vector in Chroma.
        We'll use agent_id as the unique 'document id'.
        """
        # Check if agent_id is already in the collection, if so we must delete and re-insert
        self.profiles_collection.delete(ids=[agent_id], where={"agent_id": agent_id})
        self.profiles_collection.add(
            documents=["Profile for agent " + agent_id],
            metadatas=[{"agent_id": agent_id}],
            embeddings=[profile_vec],
            ids=[agent_id]
        )

    def get_profile(self, agent_id: str) -> Optional[List[float]]:
        """
        Retrieve the agent's stored profile embedding from Chroma.
        """
        # Query the collection by ID
        results = self.profiles_collection.get(ids=[agent_id])
        if results and results["embeddings"]:
            return results["embeddings"][0]
        return None
