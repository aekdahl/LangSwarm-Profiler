"""
Models and data structures for LangProfiler.
"""

import uuid
import time

def generate_agent_id():
    """Generate a unique ID for an agent."""
    return str(uuid.uuid4())

class Agent:
    def __init__(self, name, cost, domain_tags=None, **kwargs):
        self.name = name
        self.cost = cost
        self.domain_tags = domain_tags or []

class Interaction:
    def __init__(self, agent_id, user_query, response, latency, feedback, timestamp=None, **kwargs):
        self.agent_id = agent_id
        self.user_query = user_query
        self.response = response
        self.latency = latency
        self.feedback = feedback
        self.timestamp = timestamp or time.time()
