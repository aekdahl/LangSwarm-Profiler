"""
Main interface to the LangProfiler functionality.
"""
import numpy as np

from .db import InMemoryDB
from .aggregator import SimpleAggregator
from .models import generate_agent_id, Agent, Interaction

class LangProfiler:
    def __init__(self, embedding_size=8):
        """
        Initialize the profiler with an in-memory DB and a simple aggregator.
        """
        self.db = InMemoryDB()
        self.aggregator = SimpleAggregator(embedding_size=embedding_size)

    # 1. Register Agent
    def register_agent(self, name, cost=0.0, domain_tags=None):
        """
        Registers a new agent with baseline metadata.
        Returns: agent_id (string)

        {
          "name": "GPT-4 Medical Agent",
          "description": "Handles medical QA with GPT-4 backend",
          "agent_version": "v1.2",
          "cost": 0.002,   # per token
          "domain_tags": ["medical"],
          "specialization": "Medical QA Expert",
          "instructions": "Please provide responses using simple medical terminology, referencing standard clinical guidelines where possible.",
          "backend_model": "GPT-4",
          "max_token_limit": 4096,
          "training_data_cutoff": "2023-01",
          "latency_bounds": {
            "avg_expected": 1.0,
            "max_acceptable": 3.0
          },
          "compliance_level": "HIPAA-compliant",
          "owner": "AI Ops Team",
          "creation_date": "2024-12-25T12:00:00Z",
          "tags": ["production", "priority"]
        }

        """
        agent_id = generate_agent_id()
        agent_obj = Agent(name, cost, domain_tags=domain_tags)
        self.db.add_agent(agent_id, agent_obj)
        return agent_id

    # 2. Log Interaction
    def log_interaction(self, agent_id, user_query, response, latency=0.0, feedback=0.0):
        """
        Logs an interaction for the given agent.

        {
          "agent_id": "123e4567-e89b-12d3-a456-426614174000",
          "query": "What is the best medication for a headache?",
          "response": "I am not a doctor, but typically...",
          "domain": "medical",
          "latency": 0.5,
          "cost": 0.002 * (11 + 25),  // if cost_per_token=0.002
          "feedback": 4,
          "timestamp": "2024-12-25T15:30:00Z"
        }

        """
        interaction_data = Interaction(agent_id, user_query, response, latency, feedback)
        self.db.add_interaction(vars(interaction_data))  # store as a dict
        # After logging, update the agent's profile
        self._update_agent_profile(agent_id)

    # 3. Private Method: Update Agent Profile
    def _update_agent_profile(self, agent_id):
        """
        Gathers interactions for the agent, aggregates them into a profile vector.
        """
        interactions = self.db.list_interactions(agent_id)
        if not interactions:
            return

        # Example: compute some aggregated metrics
        avg_latency = np.mean([itx['latency'] for itx in interactions])
        avg_feedback = np.mean([itx['feedback'] for itx in interactions])

        # Also pull cost/domain info from the agent object
        agent_obj = self.db.get_agent(agent_id)
        cost = agent_obj.cost if agent_obj else 0.0

        # Build a dict of metrics
        agent_data = {
            "avg_latency": avg_latency,
            "avg_cost": cost,
            "avg_feedback": avg_feedback
        }

        # Convert these metrics into an embedding
        profile_vec = self.aggregator.aggregate(agent_data)

        # Store in DB
        self.db.update_profile(agent_id, profile_vec)

    # 4. Get Current Profile
    def get_current_profile(self, agent_id):
        """
        Returns the current embedding vector for a given agent, or None if not found.
        """
        return self.db.get_profile(agent_id)

    # 5. (Optional) Additional retrieval or historical queries
    #    can be implemented here as needed.
