import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from config.settings import CHAT_DB_PATH, EMBEDDINGS_DIR

class MemoryDatabase:
    def __init__(self, db_path=None, qdrant_client=None):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path or CHAT_DB_PATH
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.qdrant_client = qdrant_client or QdrantClient(path=f"{EMBEDDINGS_DIR}/chat")
        self.init_db()

    def init_db(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    title TEXT
                )
            ''')
            
            # Create messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            ''')
            
            conn.commit()

    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation and return its ID."""
        conversation_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
             cursor = conn.cursor()
             cursor.execute(
                 'INSERT INTO conversations (conversation_id, title) VALUES (?, ?)',
                 (conversation_id, title)
             )
             conn.commit()
        return conversation_id

    def close_conversation(self, conversation_id: str) -> bool:
        """Close a conversation by setting its end time."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE conversations SET end_time = CURRENT_TIMESTAMP WHERE conversation_id = ?',
                    (conversation_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error:
            return False

    def add_message(self, conversation_id: str, role: str, message: str) -> bool:
        """Add a message to a conversation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO messages (conversation_id, role, message) VALUES (?, ?, ?)',
                    (conversation_id, role, message)
                )
                conn.commit()
                return True
        except sqlite3.Error:
            return False

    def get_messages(self, conversation_id: str) -> List[tuple[str, str, datetime]]:
        """Retrieve all messages from a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                'SELECT role, message, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp',
                (conversation_id,)
            )
            return [(row['role'], row['message'], row['timestamp']) for row in cursor.fetchall()]

    def get_conversation_history(self, limit: int = 10) -> List[dict]:
        """Retrieve recent conversations with their details."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    c.conversation_id,
                    c.start_time,
                    c.end_time,
                    c.title,
                    COUNT(m.message_id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                GROUP BY c.conversation_id
                ORDER BY c.start_time DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_conversation_title(self, conversation_id: str) -> Optional[str]:
        """Get the title of a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT title FROM conversations WHERE conversation_id = ?',
                (conversation_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update the title of a conversation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE conversations SET title = ? WHERE conversation_id = ?',
                    (title, conversation_id)
                )
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error:
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Delete messages first due to foreign key constraint
                cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
                cursor.execute('DELETE FROM conversations WHERE conversation_id = ?', (conversation_id,))
                conn.commit()
                return True
        except sqlite3.Error:
            return False

    def store_embedding_in_qdrant(self, message_id: str, embedding: List[float], metadata: dict):
        """Store the embedding in Qdrant with metadata."""
        collection_name = "chat_memory"
        
        # Ensure collection exists
        try:
            self.qdrant_client.get_collection(collection_name)
        except Exception as e:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": len(embedding), "distance": "Cosine"}
            )
            
        point = PointStruct(
            id=message_id,
            vector=embedding,
            payload=metadata
        )
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=[point]
        )

    def search_embeddings_in_qdrant(self, query_embedding: List[float], filters: dict = None) -> List[dict]:
        """Search for the most relevant messages in Qdrant."""
        collection_name = "chat_memory"
        
        try:
            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=5,
                filter=filters or {}
            )
            
            return [
                {
                    "message_id": result.id,
                    "score": result.score,
                    "metadata": result.payload
                }
                for result in search_results
            ]
        except Exception as e:
            print(f"Error searching embeddings: {e}")
            return [] 