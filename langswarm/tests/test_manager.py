def test_register_agent(profiler):
    profiler.register_agent("agent_1", {"name": "Test Agent"}, ["intent", "sentiment"])
    agent = profiler.get_agent_info("agent_1")
    assert agent is not None
    assert agent["name"] == "Test Agent"
    assert "intent" in agent["features"]

def test_log_interaction(profiler):
    profiler.register_agent("agent_1", {"name": "Test Agent"}, ["intent", "sentiment"])
    profiler.log_interaction(
        agent_id="agent_1",
        query="What is AI?",
        response="Artificial Intelligence is a branch of computer science.",
        latency=0.5,
        feedback=4.5
    )
    interactions = profiler.list_interactions("agent_1")
    assert len(interactions) == 1
    assert interactions[0]["query"] == "What is AI?"

def test_update_profile(profiler):
    profiler.register_agent("agent_1", {"name": "Test Agent"}, ["intent", "sentiment"])
    profiler.log_interaction(
        agent_id="agent_1",
        query="What is AI?",
        response="Artificial Intelligence is a branch of computer science.",
        latency=0.5,
        feedback=4.5
    )
    profiler.update_profile("agent_1")
    profile = profiler.get_current_profile("agent_1")
    assert profile is not None
    assert len(profile) > 0
