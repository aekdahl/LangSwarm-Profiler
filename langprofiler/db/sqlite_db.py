# db/sqlite_db.py
import sqlite3
import json
from typing import List, Optional
from .base import DBBase

class SqliteDB(DBBase):
    def __init__(self, db_path: str = "langprofiler.db"):
        """
        :param db_path: SQLite file path. Use ':memory:' for in-memory testing.
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    # db/sqlite_db.py (updated schema)

    def _create_tables(self):
        cur = self.conn.cursor()
    
        # ------- AGENTS -------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                agent_info TEXT,
                features TEXT  -- JSON-encoded list of features
            )
        """)
    
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                query TEXT,
                response TEXT,
                timestamp REAL,
                latency REAL,
                feedback REAL,
                features TEXT,  -- JSON-encoded dictionary of extracted features
                FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
            )
        """)
    
        cur.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                agent_id TEXT PRIMARY KEY,
                profile_vec TEXT,
                FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
            )
        """)
    
        # ------- PROMPTS -------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id TEXT PRIMARY KEY,
                prompt_info TEXT,
                features TEXT  -- JSON-encoded list of features
            )
        """)
    
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prompt_interactions (
                prompt_interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT,
                query TEXT,
                response TEXT,
                timestamp REAL,
                latency REAL,
                feedback REAL,
                features TEXT,  -- JSON-encoded dictionary of extracted features
                FOREIGN KEY(prompt_id) REFERENCES prompts(prompt_id)
            )
        """)
    
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prompt_profiles (
                prompt_id TEXT PRIMARY KEY,
                profile_vec TEXT,
                FOREIGN KEY(prompt_id) REFERENCES prompts(prompt_id)
            )
        """)
    
        self.conn.commit()

    # =======================
    #        AGENTS
    # =======================
    def add_agent(self, agent_id: str, agent_info: dict) -> None:
        cur = self.conn.cursor()
        agent_info_json = json.dumps(agent_info)
        cur.execute(
            "INSERT OR REPLACE INTO agents (agent_id, agent_info) VALUES (?, ?)",
            (agent_id, agent_info_json)
        )
        self.conn.commit()

    def get_agent(self, agent_id: str) -> Optional[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT agent_info FROM agents WHERE agent_id = ?", (agent_id,))
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def add_interaction(self, interaction_data: dict) -> None:
        """
        interaction_data fields:
            agent_id, query, response, timestamp, latency, feedback
        """
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO interactions 
            (agent_id, query, response, timestamp, latency, feedback)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            interaction_data.get("agent_id"),
            interaction_data.get("query"),
            interaction_data.get("response"),
            interaction_data.get("timestamp"),
            interaction_data.get("latency", 0.0),
            interaction_data.get("feedback", 0.0)
        ))
        self.conn.commit()

    def list_interactions(self, agent_id: str) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT agent_id, query, response, timestamp, latency, feedback
            FROM interactions
            WHERE agent_id = ?
        """, (agent_id,))
        rows = cur.fetchall()
        results = []
        for row in rows:
            results.append({
                "agent_id": row[0],
                "query": row[1],
                "response": row[2],
                "timestamp": row[3],
                "latency": row[4],
                "feedback": row[5]
            })
        return results

    def update_profile(self, agent_id: str, profile_vec: List[float]) -> None:
        cur = self.conn.cursor()
        profile_json = json.dumps(profile_vec)
        cur.execute("""
            INSERT OR REPLACE INTO profiles (agent_id, profile_vec)
            VALUES (?, ?)
        """, (agent_id, profile_json))
        self.conn.commit()

    def get_profile(self, agent_id: str) -> Optional[List[float]]:
        cur = self.conn.cursor()
        cur.execute("SELECT profile_vec FROM profiles WHERE agent_id = ?", (agent_id,))
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None

    # =======================
    #        PROMPTS
    # =======================
    def add_prompt(self, prompt_id: str, prompt_info: dict) -> None:
        cur = self.conn.cursor()
        prompt_info_json = json.dumps(prompt_info)
        cur.execute(
            "INSERT OR REPLACE INTO prompts (prompt_id, prompt_info) VALUES (?, ?)",
            (prompt_id, prompt_info_json)
        )
        self.conn.commit()

    def get_prompt(self, prompt_id: str) -> Optional[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT prompt_info FROM prompts WHERE prompt_id = ?", (prompt_id,))
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def add_prompt_interaction(self, interaction_data: dict) -> None:
        """
        interaction_data fields:
            prompt_id, query, response, timestamp, latency, feedback
        """
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO prompt_interactions
            (prompt_id, query, response, timestamp, latency, feedback)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            interaction_data.get("prompt_id"),
            interaction_data.get("query"),
            interaction_data.get("response"),
            interaction_data.get("timestamp"),
            interaction_data.get("latency", 0.0),
            interaction_data.get("feedback", 0.0)
        ))
        self.conn.commit()

    def list_prompt_interactions(self, prompt_id: str) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT prompt_id, query, response, timestamp, latency, feedback
            FROM prompt_interactions
            WHERE prompt_id = ?
        """, (prompt_id,))
        rows = cur.fetchall()
        results = []
        for row in rows:
            results.append({
                "prompt_id": row[0],
                "query": row[1],
                "response": row[2],
                "timestamp": row[3],
                "latency": row[4],
                "feedback": row[5]
            })
        return results

    def update_prompt_profile(self, prompt_id: str, profile_vec: List[float]) -> None:
        cur = self.conn.cursor()
        profile_json = json.dumps(profile_vec)
        cur.execute("""
            INSERT OR REPLACE INTO prompt_profiles (prompt_id, profile_vec)
            VALUES (?, ?)
        """, (prompt_id, profile_json))
        self.conn.commit()

    def get_prompt_profile(self, prompt_id: str) -> Optional[List[float]]:
        cur = self.conn.cursor()
        cur.execute("SELECT profile_vec FROM prompt_profiles WHERE prompt_id = ?", (prompt_id,))
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def __del__(self):
        self.conn.close()
