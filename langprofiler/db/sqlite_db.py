# db/sqlite_db.py
import sqlite3
import json
from typing import List, Optional

from .base import DBBase

class SqliteDB(DBBase):
    def __init__(self, db_path: str = "langprofiler.db"):
        """
        :param db_path: Path to the SQLite file. Using ':memory:' for in-memory DB is also possible.
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()

        # Agents table (stores JSON for agent_info)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                agent_info TEXT
            )
        """)

        # Interactions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                query TEXT,
                response TEXT,
                timestamp REAL,
                latency REAL,
                feedback REAL,
                FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
            )
        """)

        # Profiles table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                agent_id TEXT PRIMARY KEY,
                profile_vec TEXT,
                FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
            )
        """)

        self.conn.commit()

    def add_agent(self, agent_id: str, agent_info: dict) -> None:
        cur = self.conn.cursor()
        # Convert the agent_info dict to JSON
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
        cur.execute("SELECT agent_id, query, response, timestamp, latency, feedback FROM interactions WHERE agent_id = ?", (agent_id,))
        rows = cur.fetchall()

        interactions = []
        for row in rows:
            interactions.append({
                "agent_id": row[0],
                "query": row[1],
                "response": row[2],
                "timestamp": row[3],
                "latency": row[4],
                "feedback": row[5]
            })
        return interactions

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

    def __del__(self):
        self.conn.close()
