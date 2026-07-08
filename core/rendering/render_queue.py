import sqlite3
import json
import logging
import os
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RenderQueue:
    def __init__(self, db_path: str = "workspace/cache/render_queue.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS render_jobs (
                    job_id TEXT PRIMARY KEY,
                    scene_id TEXT,
                    shot_id TEXT,
                    prompt_hash TEXT,
                    image_path TEXT,
                    status TEXT,
                    attempts INTEGER DEFAULT 0,
                    seed INTEGER,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    stage TEXT,
                    message TEXT,
                    details TEXT
                )
            """)
            conn.commit()
            
    def add_job(self, job_id: str, scene_id: str, shot_id: str, prompt_hash: str, seed: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO render_jobs 
                (job_id, scene_id, shot_id, prompt_hash, status, seed, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'PENDING', ?, ?, ?)
            """, (job_id, scene_id, shot_id, prompt_hash, seed, datetime.utcnow(), datetime.utcnow()))
            
    def update_job_status(self, job_id: str, status: str, image_path: Optional[str] = None):
        with sqlite3.connect(self.db_path) as conn:
            if image_path:
                conn.execute("""
                    UPDATE render_jobs SET status = ?, image_path = ?, updated_at = ?
                    WHERE job_id = ?
                """, (status, image_path, datetime.utcnow(), job_id))
            else:
                conn.execute("""
                    UPDATE render_jobs SET status = ?, updated_at = ?
                    WHERE job_id = ?
                """, (status, datetime.utcnow(), job_id))
                
    def increment_attempts(self, job_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE render_jobs SET attempts = attempts + 1, updated_at = ?
                WHERE job_id = ?
            """, (datetime.utcnow(), job_id))
            
    def get_all_jobs(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM render_jobs")
            return [dict(row) for row in cursor.fetchall()]

    def get_pending_jobs(self, limit: int = 10) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM render_jobs 
                WHERE status IN ('PENDING', 'FAILED') AND attempts < 3
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
            
    def log_event(self, stage: str, message: str, details: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO event_log (timestamp, stage, message, details)
                VALUES (?, ?, ?, ?)
            """, (datetime.utcnow(), stage, message, details))
            
    def vacuum(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
