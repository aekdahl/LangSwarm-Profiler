"""
usage_prompts.py

Example usage of LangProfiler focusing on PROMPTS.
"""
import os
from langprofiler.config import ProfilerConfig
from langprofiler.manager import LangProfiler

def main():
    # 1. Configure environment for SQLite (default)
    os.environ["PROFILER_DB_BACKEND"] = "sqlite"
    os.environ["PROFILER_DB_DSN"] = "prompt_example.db"

    # 2. Initialize the profiler
    config = ProfilerConfig()
    profiler = LangProfiler(config)

    # 3. Register a prompt
    prompt_id = "prompt-001"
    prompt_info = {
        "text": "You are a friendly chatbot. Greet the user politely.",
        "numeric_features": [0.2, 0.7],  # e.g., 'importance' and 'creativity' rating
    }
    profiler.register_prompt(prompt_id, prompt_info)
    print(f"Registered prompt '{prompt_id}'")

    # 4. Log interactions for this prompt
    profiler.log_prompt_interaction(
        prompt_id=prompt_id,
        query="Say hello in a friendly manner",
        response="Hello there! How can I help you today?",
        latency=0.2,
        feedback=4.5
    )
    profiler.log_prompt_interaction(
        prompt_id=prompt_id,
        query="Greet a new user with a warm introduction",
        response="Welcome aboard! It's great to have you here. How can I assist?",
        latency=0.25,
        feedback=4.0
    )
    print(f"Logged 2 interactions for prompt '{prompt_id}'")

    # 5. Get the prompt's current profile
    profile_vec = profiler.get_prompt_profile(prompt_id)
    print(f"Profile for prompt '{prompt_id}':\n", profile_vec)

    # 6. List all interactions for this prompt
    interactions = profiler.list_prompt_interactions(prompt_id)
    print(f"Interactions for prompt '{prompt_id}':\n", interactions)

if __name__ == "__main__":
    main()
