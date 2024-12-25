"""
usage_agents.py

Example usage of LangProfiler focusing on AGENTS.
"""
import os
from langprofiler.config import ProfilerConfig
from langprofiler.manager import LangProfiler

def main():
    # 1. Configure environment for SQLite (default)
    os.environ["PROFILER_DB_BACKEND"] = "sqlite"  
    os.environ["PROFILER_DB_DSN"] = "agent_example.db"

    # 2. Initialize the profiler
    config = ProfilerConfig()
    profiler = LangProfiler(config)

    # 3. Register an agent
    agent_id = "agent-abc"
    agent_info = {
        "name": "GPT-4 Medical Agent",
        "cost": 0.003,
        "instructions": "Provide medical advice in a helpful, safe manner",
        "numeric_features": [0.003, 0.9]  # Example: cost, domain expertise
    }
    profiler.register_agent(agent_id, agent_info)
    print(f"Registered agent '{agent_id}'")

    # 4. Log interactions for this agent
    profiler.log_interaction(
        agent_id=agent_id,
        query="What is the best medication for a headache?",
        response="Typically NSAIDs like Ibuprofen can help...",
        latency=0.5,
        feedback=5.0
    )
    profiler.log_interaction(
        agent_id=agent_id,
        query="Could you tell me more about migraines?",
        response="Migraines often require prescription medication...",
        latency=0.6,
        feedback=4.5
    )
    print(f"Logged 2 interactions for agent '{agent_id}'")

    # 5. Get the agent's current profile
    profile_vec = profiler.get_current_profile(agent_id)
    print(f"Profile for agent '{agent_id}':\n", profile_vec)

    # 6. List all interactions for this agent
    interactions = profiler.list_interactions(agent_id)
    print(f"Interactions for agent '{agent_id}':\n", interactions)

if __name__ == "__main__":
    main()
