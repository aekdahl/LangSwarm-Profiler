# LangSwarm-Profiler

LangSwarm-Profiler is a Python library designed to profile and analyze the performance of Large Language Model (LLM) agents. By collecting and processing data from interactions, LangSwarm-Profiler generates embeddings that represent an agent's capabilities, making it easier to evaluate, compare, and improve multi-agent systems.

## Features

- **Agent Registration**: Register LLM agents with detailed metadata and feature tracking.
- **Interaction Logging**: Log queries, responses, and associated metrics for performance analysis.
- **Profile Embedding Generation**: Create fixed-size embeddings for agents based on their interactions and attributes.
- **Feature Extraction**: Analyze text inputs to derive insights such as sentiment, intent, and topic.
- **Integration with Ecosystem Tools**:
  - **LangSwarm**: Use LangSwarm-Profiler's data to aid in multi-agent consensus and decision-making.
  - **MemSwarm**: Store and share agent profiles across systems.
  - **LangRL**: Enable reinforcement learning workflows with informed agent selection.

## Installation

Install LangSwarm-Profiler using pip:

```bash
pip install langprofiler
```

## Quick Start

Here’s a simple example to get started with LangSwarm-Profiler:

```python
from langprofiler import LangSwarm-Profiler

# Initialize the profiler
lp = LangSwarm-Profiler()

# Register an agent
agent_id = lp.register_agent(name="Test Agent", cost=0.001)

# Log an interaction
lp.log_interaction(agent_id, "What is the capital of France?", "Paris", latency=0.2, feedback=5.0)

# Retrieve the current profile
profile = lp.get_current_profile(agent_id)
print(profile)
```

## Core Concepts

### Agent Profiling
An "agent" can represent an LLM or a specialized persona within an LLM. LangSwarm-Profiler maintains profiles for each agent, capturing:
- **Performance Metrics**: Latency, accuracy, feedback scores.
- **Domain Expertise**: Tags indicating areas of specialization (e.g., medical, financial).
- **Cost Efficiency**: Estimated costs associated with the agent's usage.

### Feature Extraction
LangSwarm-Profiler can extract the following features from text:
- **Intent**: Determine the purpose of a query.
- **Sentiment**: Analyze the emotional tone of text.
- **Entities**: Identify named entities using NLP.
- **Syntax Complexity**: Measure structural complexity.
- **Key Phrases**: Highlight important terms or phrases.

### Data Pipeline
1. **Data Ingestion**: Log agent interactions.
2. **Feature Extraction**: Derive structured metrics from raw text inputs.
3. **Profile Embedding**: Use neural networks or other methods to generate fixed-size embeddings.
4. **Querying**: Access agent profiles for analysis or decision-making.

## Integration

LangSwarm-Profiler is designed to integrate seamlessly into multi-agent ecosystems. Examples:

- **LangSwarm**: Use LangSwarm-Profiler data to evaluate agent performance during consensus-based workflows.
- **MemSwarm**: Store and retrieve agent profiles across collaborative systems.
- **LangRL**: Inform reinforcement learning workflows with detailed agent embeddings.

## Example Use Case

Imagine building a multi-agent system for answering medical queries. You register two agents powered by different LLMs, log their interactions with a test dataset, and use LangSwarm-Profiler to:
1. Compare their performance metrics.
2. Select the best agent for a given domain.
3. Track improvements over time as models are updated.

```python
# Example: Comparing two agents
agent1 = lp.register_agent(name="MedicalAgent1", cost=0.002)
agent2 = lp.register_agent(name="MedicalAgent2", cost=0.003)

lp.log_interaction(agent1, "What are the symptoms of diabetes?", "Thirst, fatigue, etc.", feedback=4.5)
lp.log_interaction(agent2, "What are the symptoms of diabetes?", "Increased thirst, hunger, and fatigue.", feedback=4.8)

profile1 = lp.get_current_profile(agent1)
profile2 = lp.get_current_profile(agent2)

# Decide which agent performs better
better_agent = agent1 if profile1['feedback'] > profile2['feedback'] else agent2
```

## Development

### Requirements
LangSwarm-Profiler requires Python 3.8 or newer. Additional dependencies are listed in `requirements.txt`.

### Running Tests
Run unit tests using `pytest`:

```bash
pytest tests/
```

### Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description of your changes.

## License

LangSwarm-Profiler is licensed under the MIT License. See the `LICENSE` file for details.

---

LangSwarm-Profiler helps you unlock the full potential of your LLM ecosystem by offering robust profiling and analytics tools. Start building smarter, more efficient multi-agent systems today!

