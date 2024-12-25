# demo_usage.py
import os
from langprofiler.manager import LangProfiler
from langprofiler.config import ProfilerConfig

def main():
    # Example: override some config via environment or directly
    os.environ["PROFILER_DB_BACKEND"] = "sqlite"  # or "chroma" or "custom_sql"
    os.environ["AGGREGATOR_NUMERIC_DIM"] = "2"

    # Initialize config and profiler
    config = ProfilerConfig()
    profiler = LangProfiler(config)

    # 1. Register an agent
    agent_id = "agent-123"
    agent_info = {
        "name": "GPT-4 Medical Expert",
        "cost": 0.002,
        "instructions": "I specialize in medical domain. Provide detailed, safe answers.",
        # example numeric features: [cost, domain_expertise_score]
        "numeric_features": [0.002, 0.9]  
    }
    profiler.register_agent(agent_id, agent_info)

    # 2. Log some interactions
    profiler.log_interaction(
        agent_id=agent_id,
        query="What is the best medication for headache?",
        response="Typically, over-the-counter NSAIDs such as ibuprofen can be recommended...",
        latency=0.6,
        feedback=5.0
    )
    profiler.log_interaction(
        agent_id=agent_id,
        query="How about migraines?",
        response="Migraines may require prescription medications like triptans...",
        latency=0.8,
        feedback=4.5
    )

    # 3. Get current profile
    profile_vec = profiler.get_current_profile(agent_id)
    print(f"Profile embedding for {agent_id}:\n", profile_vec)

    # 4. List interactions
    interactions = profiler.list_interactions(agent_id)
    print(f"Interactions for {agent_id}:\n", interactions)

if __name__ == "__main__":
    main()
