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
        
        # Whether to combine numeric features
        self.AGGREGATOR_NUMERIC_DIM = int(os.getenv("AGGREGATOR_NUMERIC_DIM", "0"))

        # Additional features dimensions
        self.AGGREGATOR_ADDITIONAL_FEATURE_DIM = int(os.getenv("AGGREGATOR_ADDITIONAL_FEATURE_DIM", "15"))

        # Final embedding size if we do dimension reduction
        self.AGGREGATOR_FINAL_DIM = int(os.getenv("AGGREGATOR_FINAL_DIM", "128"))
        

        # Example: A separate "custom_sql" init script or schema path
        self.CUSTOM_SQL_INIT_DDL = os.getenv("CUSTOM_SQL_INIT_DDL", "schema.sql")

        # Feature order configuration
        self.FEATURE_ORDER = os.getenv("FEATURE_ORDER", "intent,sentiment,topic,entities,summarization,syntax_complexity,readability_score,tone,formality_level,politeness,contextual_coherence,key_phrase_extraction,temporal_features,speech_acts,intent_confidence_scores").split(",")  
