# db/custom_sql.py
from typing import List, Optional
import json
import time
import sqlite3  # or psycopg2, pyodbc, MySQLdb, etc.

from .base import DBBase

class CustomSQLConnector(DBBase):
    """
    A generic SQL-based connector. 
    Users can provide their own DSN or connection parameters.
    """

    def __init__(self, dsn: str, init_ddl: Optional[str] = None):
        """
        :param dsn: A connection string or DSN for your SQL engine.
        :param init_ddl: Optional path to a SQL file or string that creates 
                         the required tables if they don't exist.
        """
        # Example: DSN could be "postgres://user:pass@localhost:5432/langprofiler"
        # or "mysql://...", or an ODBC DSN.
        # For demonstration, we'll just reuse sqlite3 to show the approach.
        # In real life, you'd parse DSN or use a library appropriate for your SQL engine.
        
        # This is for demonstration: if DSN is a file, we treat it as a SQLite file:
        self.conn = sqlite3.connect(dsn, check_same_thread=False)
        
        if init_ddl:
            self._init_db(init_ddl)

    def _init_db(self, init_ddl: str):
        """
        Initialize DB schema from a given DDL script.
        """
        cur = self.conn.cursor()
        if init_ddl.endswith(".sql"):
            # If it's a file path, read it
            with open(init_ddl, "r") as f:
                sql_script = f.read()
        else:
            # Otherwise, assume it's raw SQL text
            sql_script = init_ddl
        cur.executescript(sql_script)
        self.conn.commit()

    def add_agent(self, agent_id: str, agent_info: dict) -> None:
        cur = self.conn.cursor()
        agent_info_json = json.dumps(agent_info)
        # Example query, adapt to your schema
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
            interaction_data.get("timestamp", time.time()),
            interaction_data.get("latency", 0.0),
            interaction_data.get("feedback", 0.0)
        ))
        self.conn.commit()

    def list_interactions(self, agent_id: str) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT agent_id, query, response, timestamp, latency, feedback 
            FROM interactions WHERE agent_id = ?
        """, (agent_id,))
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
