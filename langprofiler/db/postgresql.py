import psycopg2
from .base import DBBase
import json

# postgresql.py

class PostgresQLConnector(DBBase):
    def __init__(self, dsn: str, init_ddl: str = None):
        self.conn = psycopg2.connect(dsn)
        if init_ddl:
            self._execute_init_ddl(init_ddl)
        self._create_tables()

    def _execute_init_ddl(self, ddl: str):
        cursor = self.conn.cursor()
        cursor.execute(ddl)
        self.conn.commit()

    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Create agents table without removed features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                agent_info JSONB,
                features JSONB,
                length_of_prompt INTEGER,
                conciseness REAL
            );
        """)
        
        # Create prompts table without removed features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id TEXT PRIMARY KEY,
                prompt_info JSONB,
                features JSONB,
                length_of_prompt INTEGER,
                conciseness REAL
            );
        """)
        
        # Create interactions table without removed features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id SERIAL PRIMARY KEY,
                agent_id TEXT REFERENCES agents(agent_id),
                query TEXT,
                response TEXT,
                timestamp TIMESTAMP,
                latency REAL,
                feedback REAL,
                features JSONB,
                length_of_prompt INTEGER,
                conciseness REAL
            );
        """)
        
        # Create prompt_interactions table without removed features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_interactions (
                prompt_interaction_id SERIAL PRIMARY KEY,
                prompt_id TEXT REFERENCES prompts(prompt_id),
                query TEXT,
                response TEXT,
                timestamp TIMESTAMP,
                latency REAL,
                feedback REAL,
                features JSONB,
                length_of_prompt INTEGER,
                conciseness REAL
            );
        """)
        
        # Create profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id SERIAL PRIMARY KEY,
                agent_id TEXT REFERENCES agents(agent_id),
                profile_vec JSONB
            );
        """)
        
        # Create prompt_profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_profiles (
                profile_id SERIAL PRIMARY KEY,
                prompt_id TEXT REFERENCES prompts(prompt_id),
                profile_vec JSONB
            );
        """)
        
        self.conn.commit()

    # Implement other DBBase abstract methods here
    # e.g., add_agent, get_agent, add_interaction, etc.

    def add_agent(self, agent_id: str, agent_info: str, features: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO agents (agent_id, agent_info, features, length_of_prompt, conciseness)
            VALUES (%s, %s, %s, %s, %s)
        """, (agent_id, agent_info, features, 0, 0.0))
        self.conn.commit()

    def get_agent(self, agent_id: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM agents WHERE agent_id = %s
        """, (agent_id,))
        return cursor.fetchone()

    def add_interaction(self, interaction_data: dict):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO interactions (agent_id, query, response, timestamp, latency, feedback, features, length_of_prompt, conciseness)
            VALUES (%s, %s, %s, to_timestamp(%s), %s, %s, %s, %s, %s)
        """, (
            interaction_data["agent_id"],
            interaction_data["query"],
            interaction_data["response"],
            interaction_data["timestamp"],
            interaction_data["latency"],
            interaction_data["feedback"],
            interaction_data["features"],
            interaction_data.get("length_of_prompt", 0),
            interaction_data.get("conciseness", 0.0)
        ))
        self.conn.commit()

    def list_interactions(self, agent_id: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM interactions WHERE agent_id = %s
        """, (agent_id,))
        return cursor.fetchall()

    def update_profile(self, agent_id: str, profile_vec: List[float]):
        cursor = self.conn.cursor()
        profile_json = json.dumps(profile_vec)
        cursor.execute("""
            INSERT INTO profiles (agent_id, profile_vec)
            VALUES (%s, %s)
            ON CONFLICT (agent_id) DO UPDATE SET profile_vec = EXCLUDED.profile_vec
        """, (agent_id, profile_json))
        self.conn.commit()

    def get_profile(self, agent_id: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM profiles WHERE agent_id = %s
        """, (agent_id,))
        return cursor.fetchone()

    def add_prompt(self, prompt_id: str, prompt_info: str, features: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO prompts (prompt_id, prompt_info, features, length_of_prompt, conciseness)
            VALUES (%s, %s, %s, %s, %s)
        """, (prompt_id, prompt_info, features, 0, 0.0))
        self.conn.commit()

    def get_prompt(self, prompt_id: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM prompts WHERE prompt_id = %s
        """, (prompt_id,))
        return cursor.fetchone()

    def add_prompt_interaction(self, interaction_data: dict):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO prompt_interactions (prompt_id, query, response, timestamp, latency, feedback, features, length_of_prompt, conciseness)
            VALUES (%s, %s, %s, to_timestamp(%s), %s, %s, %s, %s, %s)
        """, (
            interaction_data["prompt_id"],
            interaction_data["query"],
            interaction_data["response"],
            interaction_data["timestamp"],
            interaction_data["latency"],
            interaction_data["feedback"],
            interaction_data["features"],
            interaction_data.get("length_of_prompt", 0),
            interaction_data.get("conciseness", 0.0)
        ))
        self.conn.commit()

    def list_prompt_interactions(self, prompt_id: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM prompt_interactions WHERE prompt_id = %s
        """, (prompt_id,))
        return cursor.fetchall()

    def update_prompt_profile(self, prompt_id: str, profile_vec: List[float]):
        cursor = self.conn.cursor()
        profile_json = json.dumps(profile_vec)
        cursor.execute("""
            INSERT INTO prompt_profiles (prompt_id, profile_vec)
            VALUES (%s, %s)
            ON CONFLICT (prompt_id) DO UPDATE SET profile_vec = EXCLUDED.profile_vec
        """, (prompt_id, profile_json))
        self.conn.commit()
