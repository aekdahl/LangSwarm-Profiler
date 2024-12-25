# config.py
import os

class ProfilerConfig:
    """
    Simple configuration class for the profiler.
    In a larger system, you might load from a .env or .yaml file, or parse CLI args.
    """

    def __init__(self):
        # Database backend: 'sqlite', 'chroma', 'custom_sql', etc.
        self.DB_BACKEND = os.getenv("PROFILER_DB_BACKEND", "sqlite")
        
        # Path or DSN for the database
        # e.g. "langprofiler.db" for SQLite, or "postgres://user:pass@localhost:5432/langprof" for custom SQL
        self.DB_DSN = os.getenv("PROFILER_DB_DSN", "langprofiler.db")

        # If using ChromaDB
        self.CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

        # Aggregator settings
        # Model name for Sentence-BERT or any other huggingface-based model
        self.AGGREGATOR_MODEL_NAME = os.getenv("AGGREGATOR_MODEL_NAME", "all-MiniLM-L6-v2")
        # Final embedding size if we do dimension reduction
        self.AGGREGATOR_FINAL_DIM = int(os.getenv("AGGREGATOR_FINAL_DIM", "32"))

        # Whether to combine numeric features
        self.AGGREGATOR_NUMERIC_DIM = int(os.getenv("AGGREGATOR_NUMERIC_DIM", "0"))

        # Example: A separate "custom_sql" init script or schema path
        self.CUSTOM_SQL_INIT_DDL = os.getenv("CUSTOM_SQL_INIT_DDL", "schema.sql")
