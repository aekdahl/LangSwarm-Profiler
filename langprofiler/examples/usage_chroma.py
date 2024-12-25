"""
usage_chroma.py

Demonstrates using ChromaDB as the backend.
We'll register both an agent and a prompt, log interactions, and retrieve profiles.
"""
import os
from langprofiler.config import ProfilerConfig
from langprofiler.manager import LangProfiler

def main():
    # 1. Set environment variables for Chroma
    os.environ["PROFILER_DB_BACKEND"] = "chroma"
    os.environ["CHROMA_PERSIST_DIR"] = "./chroma_data"  # Directory for Chroma storage

    # 2. Initialize the profiler
    config = ProfilerConfig()
    profiler = LangProfiler(config)

    # 3. Register an agent
    agent_id = "agent-chroma"
    agent_info = {
        "name": "Finance GPT",
        "instructions": "Provide financial advice with disclaimers",
        "numeric_features": [0.005, 0.8],  # cost, domain expertise
    }
    profiler.register_agent(agent_id, agent_info)
    print(f"Registered agent '{agent_id}' using ChromaDB backend")

    # 4. Log agent interaction
    profiler.log_interaction(
        agent_id=agent_id,
        query="Should I invest in stocks or bonds?",
        response="It depends on your risk tolerance. Diversifying is recommended...",
        latency=0.3,
        feedback=4.0
    )

    # 5. Retrieve agent profile from Chroma
    agent_profile = profiler.get_current_profile(agent_id)
    print(f"Chroma-based profile for agent '{agent_id}':\n", agent_profile)

    # 6. Register a prompt
    prompt_id = "prompt-chroma"
    prompt_info = {
        "text": "You are a cooking assistant. Provide recipes in a step-by-step format.",
        "numeric_features": [0.2, 0.9],  # Some arbitrary features
    }
    profiler.register_prompt(prompt_id, prompt_info)
    print(f"Registered prompt '{prompt_id}' using ChromaDB backend")

    # 7. Log prompt interaction
    profiler.log_prompt_interaction(
        prompt_id=prompt_id,
        query="Share a simple pasta recipe",
        response="Sure! 1) Boil water. 2) Add pasta. 3) Cook 8 min. ...",
        latency=0.2,
        feedback=4.5
    )

    # 8. Retrieve prompt profile from Chroma
    prompt_profile = profiler.get_prompt_profile(prompt_id)
    print(f"Chroma-based profile for prompt '{prompt_id}':\n", prompt_profile)

if __name__ == "__main__":
    main()
