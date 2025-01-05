from langswarm.profiler.config import ProfilerConfig

def test_default_config():
    config = ProfilerConfig()
    assert config.DB_BACKEND == "sqlite"
    assert config.AGGREGATOR_MODEL_NAME == "all-MiniLM-L6-v2"

def test_env_override(monkeypatch):
    monkeypatch.setenv("PROFILER_DB_BACKEND", "chroma")
    config = ProfilerConfig()
    assert config.DB_BACKEND == "chroma"
