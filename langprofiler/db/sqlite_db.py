# sqlite_db.py

import sqlite3
from .base import DBBase

class SqliteDB(DBBase):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Create agents table without removed features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                agent_info TEXT,
                features TEXT,
                length_of_prompt INTEGER,
                conciseness REAL
            );
        """)
        
        # Create prompts table without removed features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id TEXT PRIMARY KEY,
                prompt_info TEXT,
                features TEXT,
                length_of_prompt INTEGER,
                conciseness REAL
            );
        """)
        
        # Create interactions table without removed features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                query TEXT,
                response TEXT,
                timestamp REAL,
                latency REAL,
                feedback REAL,
                features TEXT,
                length_of_prompt INTEGER,
                conciseness REAL,
                FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
            );
        """)
        
        # Create prompt_interactions table without removed features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_interactions (
                prompt_interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT,
                query TEXT,
                response TEXT,
                timestamp REAL,
                latency REAL,
                feedback REAL,
                features TEXT,
                length_of_prompt INTEGER,
                conciseness REAL,
                FOREIGN KEY(prompt_id) REFERENCES prompts(prompt_id)
            );
        """)
        
        # Create profiles table (assuming profiles are stored separately)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                profile_vec TEXT,
                FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_profiles (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT,
                profile_vec TEXT,
                FOREIGN KEY(prompt_id) REFERENCES prompts(prompt_id)
            );
        """)
        
        self.conn.commit()

    # Implement other DBBase abstract methods here
    # e.g., add_agent, get_agent, add_interaction, etc.

    def add_agent(self, agent_id: str, agent_info: str, features: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO agents (agent_id, agent_info, features, length_of_prompt, conciseness)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_id, agent_info, features, 0, 0.0))
        self.conn.commit()

    def get_agent(self, agent_id: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM agents WHERE agent_id = ?
        """, (agent_id,))
        return cursor.fetchone()

    def add_interaction(self, interaction_data: dict):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO interactions (agent_id, query, response, timestamp, latency, feedback, features, length_of_prompt, conciseness)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            SELECT * FROM interactions WHERE agent_id = ?
        """, (agent_id,))
        return cursor.fetchall()

    def update_profile(self, agent_id: str, profile_vec: List[float]):
        cursor = self.conn.cursor()
        profile_json = json.dumps(profile_vec)
        cursor.execute("""
            INSERT INTO profiles (agent_id, profile_vec)
            VALUES (?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET profile_vec=excluded.profile_vec
        """, (agent_id, profile_json))
        self.conn.commit()

    def get_profile(self, agent_id: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM profiles WHERE agent_id = ?
        """, (agent_id,))
        return cursor.fetchone()

    def add_prompt(self, prompt_id: str, prompt_info: str, features: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO prompts (prompt_id, prompt_info, features, length_of_prompt, conciseness)
            VALUES (?, ?, ?, ?, ?)
        """, (prompt_id, prompt_info, features, 0, 0.0))
        self.conn.commit()

    def get_prompt(self, prompt_id: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM prompts WHERE prompt_id = ?
        """, (prompt_id,))
        return cursor.fetchone()

    def add_prompt_interaction(self, interaction_data: dict):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO prompt_interactions (prompt_id, query, response, timestamp, latency, feedback, features, length_of_prompt, conciseness)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            SELECT * FROM prompt_interactions WHERE prompt_id = ?
        """, (prompt_id,))
        return cursor.fetchall()

    def update_prompt_profile(self, prompt_id: str, profile_vec: List[float]):
        cursor = self.conn.cursor()
        profile_json = json.dumps(profile_vec)
        cursor.execute("""
            INSERT INTO prompt_profiles (prompt_id, profile_vec)
            VALUES (?, ?)
            ON CONFLICT(prompt_id) DO UPDATE SET profile_vec=excluded.profile_vec
        """, (prompt_id, profile_json))
        self.conn.commit()
