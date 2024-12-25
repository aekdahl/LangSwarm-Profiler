# db/chroma_db.py
from typing import List, Optional
from .base import DBBase

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(
        "ChromaDB is not installed.\n"
        "Please install it with: pip install chromadb\n"
        "Error: " + str(e)
    )

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
