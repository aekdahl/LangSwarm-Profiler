def test_add_and_get_agent(db):
    db.add_agent("agent_1", {"name": "Test Agent"}, [])
    agent = db.get_agent("agent_1")
    assert agent is not None
    assert agent["name"] == "Test Agent"

def test_add_interaction(db):
    db.add_agent("agent_1", {"name": "Test Agent"}, [])
    interaction = {
        "agent_id": "agent_1",
        "query": "What is AI?",
        "response": "AI stands for Artificial Intelligence.",
        "timestamp": 1234567890.0,
        "latency": 0.5,
        "feedback": 5,
        "features": {"intent": "educational inquiry"}
    }
    db.add_interaction(interaction)
    interactions = db.list_interactions("agent_1")
    assert len(interactions) == 1
    assert interactions[0]["query"] == "What is AI?"
