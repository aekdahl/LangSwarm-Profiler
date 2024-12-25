"""
Database / Storage handling for LangProfiler.
For the first version, we'll use an in-memory dict as a placeholder.
Later, you can swap to a real DB (e.g., SQLite, PostgreSQL).
"""

class InMemoryDB:
    def __init__(self):
        # Agents, interactions, and profiles stored in dictionaries
        self.agents = {}         # agent_id -> agent_info dict
        self.interactions = []   # list of interaction dicts
        self.profiles = {}       # agent_id -> latest_profile_vec

    def add_agent(self, agent_id, agent_info):
        self.agents[agent_id] = agent_info

    def get_agent(self, agent_id):
        return self.agents.get(agent_id, None)

    def add_interaction(self, interaction_data):
        self.interactions.append(interaction_data)

    def update_profile(self, agent_id, profile_vec):
        self.profiles[agent_id] = profile_vec

    def get_profile(self, agent_id):
        return self.profiles.get(agent_id, None)

    def list_interactions(self, agent_id):
        # Return all interactions for a given agent
        return [itx for itx in self.interactions if itx["agent_id"] == agent_id]
