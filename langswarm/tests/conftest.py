import pytest
from langswarm.profiler.manager import LangProfiler
from langswarm.profiler.config import ProfilerConfig
from langswarm.profiler.db.in_memory import InMemoryDB

@pytest.fixture
def config():
    return ProfilerConfig()

@pytest.fixture
def db():
    return InMemoryDB()

@pytest.fixture
def profiler(config, db, monkeypatch):
    # Monkeypatch to replace DB with in-memory DB
    monkeypatch.setattr("langswarm.profiler.manager.LangProfiler._init_db", lambda self: db)
    return LangProfiler(config=config)
