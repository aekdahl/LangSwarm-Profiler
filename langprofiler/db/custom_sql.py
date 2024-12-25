# db/custom_sql.py
from typing import List, Optional
import json
import time
import sqlite3  # or psycopg2, pyodbc, etc.
from .base import DBBase

class CustomSQLConnector(DBBase):
    """
    Generic SQL-based connector for both Agents and Prompts.
    The DSN can point to any DB-API–compatible database.
    """

    def __init__(self, dsn: str, init_ddl: Optional[str] = None):
        """
        :param dsn: Connection string or file path. E.g. "langprofiler.db" or a Postgres DSN.
        :param init_ddl: Optional path to .sql file or a raw SQL string that creates the needed tables.
        """
        self.conn = sqlite3.connect(dsn, check_same_thread=False)
        if init_ddl:
            self._init_db(init_ddl)

    def _init_db(self, init_ddl: str):
        cur = self.conn.cursor()
        if init_ddl.endswith(".sql"):
            with open(init_ddl, "r") as f:
                sql_script = f.read()
        else:
            sql_script = init_ddl
        cur.executescript(sql_script)
        self.conn.commit()

    # ========== AGENTS ==========
    def add_agent(self, agent_id: str, agent_info: dict) -> None:
        cur = self.conn.cursor()
        agent_info_json = json.dumps(agent_info)
        cur.execute("""
            INSERT OR REPLACE INTO agents (agent_id, agent_info) 
            VALUES (?, ?)
        """, (agent_id, agent_info_json))
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

    # ========== PROMPTS ==========
    def add_prompt(self, prompt_id: str, prompt_info: dict) -> None:
        cur = self.conn.cursor()
        prompt_info_json = json.dumps(prompt_info)
        cur.execute("""
            INSERT OR REPLACE INTO prompts (prompt_id, prompt_info)
            VALUES (?, ?)
        """, (prompt_id, prompt_info_json))
        self.conn.commit()

    def get_prompt(self, prompt_id: str) -> Optional[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT prompt_info FROM prompts WHERE prompt_id = ?", (prompt_id,))
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def add_prompt_interaction(self, interaction_data: dict) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO prompt_interactions
            (prompt_id, query, response, timestamp, latency, feedback)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            interaction_data.get("prompt_id"),
            interaction_data.get("query"),
            interaction_data.get("response"),
            interaction_data.get("timestamp", time.time()),
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
