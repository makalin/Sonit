"""
Database manager module for Sonit
Handles storage and retrieval of training data and user samples
"""

import sqlite3
import os
import json
import numpy as np
import pickle
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any


class DatabaseManager:
    """SQLite database manager for Sonit application"""
    
    def __init__(self, db_path='data/sonit.db'):
        """
        Initialize database manager
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.ensure_db_directory()
        self.init_database()
    
    def ensure_db_directory(self):
        """Ensure database directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create samples table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    features BLOB NOT NULL,
                    label TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT DEFAULT 'default',
                    confidence REAL DEFAULT 0.0,
                    metadata TEXT
                )
            ''')
            
            # Create models table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    accuracy REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Create training_sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    num_samples INTEGER,
                    accuracy REAL,
                    duration REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            ''')
            
            # Create settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_samples_label ON samples (label)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_samples_user ON samples (user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_samples_created ON samples (created_at)')
            
            conn.commit()
    
    def save_sample(self, features: np.ndarray, label: str, user_id: str = 'default', 
                   confidence: float = 0.0, metadata: Dict[str, Any] = None) -> int:
        """
        Save a training sample to the database
        
        Args:
            features (np.ndarray): Feature vector
            label (str): Sample label
            user_id (str): User identifier
            confidence (float): Confidence score
            metadata (dict): Additional metadata
            
        Returns:
            int: Sample ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Serialize features
            features_blob = pickle.dumps(features)
            
            # Serialize metadata
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute('''
                INSERT INTO samples (features, label, user_id, confidence, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (features_blob, label, user_id, confidence, metadata_json))
            
            sample_id = cursor.lastrowid
            conn.commit()
            
            return sample_id
    
    def get_sample(self, sample_id: int) -> Optional[Tuple[int, np.ndarray, str, str, float, Dict[str, Any]]]:
        """
        Get a sample by ID
        
        Args:
            sample_id (int): Sample ID
            
        Returns:
            tuple: (id, features, label, user_id, confidence, metadata) or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, features, label, user_id, confidence, metadata
                FROM samples WHERE id = ?
            ''', (sample_id,))
            
            row = cursor.fetchone()
            
            if row:
                sample_id, features_blob, label, user_id, confidence, metadata_json = row
                features = pickle.loads(features_blob)
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                return sample_id, features, label, user_id, confidence, metadata
            
            return None
    
    def get_all_samples(self, user_id: str = None) -> List[Tuple[int, np.ndarray, str]]:
        """
        Get all samples
        
        Args:
            user_id (str): Filter by user ID (optional)
            
        Returns:
            list: List of (id, features, label) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('''
                    SELECT id, features, label FROM samples 
                    WHERE user_id = ? ORDER BY created_at
                ''', (user_id,))
            else:
                cursor.execute('''
                    SELECT id, features, label FROM samples 
                    ORDER BY created_at
                ''')
            
            rows = cursor.fetchall()
            
            samples = []
            for row in rows:
                sample_id, features_blob, label = row
                features = pickle.loads(features_blob)
                samples.append((sample_id, features, label))
            
            return samples
    
    def get_samples_by_label(self, label: str, user_id: str = None) -> List[Tuple[int, np.ndarray, str]]:
        """
        Get samples by label
        
        Args:
            label (str): Label to filter by
            user_id (str): Filter by user ID (optional)
            
        Returns:
            list: List of (id, features, label) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('''
                    SELECT id, features, label FROM samples 
                    WHERE label = ? AND user_id = ? ORDER BY created_at
                ''', (label, user_id))
            else:
                cursor.execute('''
                    SELECT id, features, label FROM samples 
                    WHERE label = ? ORDER BY created_at
                ''', (label,))
            
            rows = cursor.fetchall()
            
            samples = []
            for row in rows:
                sample_id, features_blob, label = row
                features = pickle.loads(features_blob)
                samples.append((sample_id, features, label))
            
            return samples
    
    def delete_sample(self, sample_id: int) -> bool:
        """
        Delete a sample by ID
        
        Args:
            sample_id (int): Sample ID to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM samples WHERE id = ?', (sample_id,))
            conn.commit()
            
            return cursor.rowcount > 0
    
    def clear_all_samples(self, user_id: str = None) -> int:
        """
        Clear all samples
        
        Args:
            user_id (str): Clear only samples for specific user (optional)
            
        Returns:
            int: Number of samples deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('DELETE FROM samples WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('DELETE FROM samples')
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            return deleted_count
    
    def get_sample_count(self, user_id: str = None) -> int:
        """
        Get total number of samples
        
        Args:
            user_id (str): Count samples for specific user (optional)
            
        Returns:
            int: Number of samples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('SELECT COUNT(*) FROM samples WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('SELECT COUNT(*) FROM samples')
            
            return cursor.fetchone()[0]
    
    def get_label_counts(self, user_id: str = None) -> Dict[str, int]:
        """
        Get count of samples per label
        
        Args:
            user_id (str): Count samples for specific user (optional)
            
        Returns:
            dict: Dictionary mapping labels to counts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('''
                    SELECT label, COUNT(*) FROM samples 
                    WHERE user_id = ? GROUP BY label
                ''', (user_id,))
            else:
                cursor.execute('''
                    SELECT label, COUNT(*) FROM samples 
                    GROUP BY label
                ''')
            
            return dict(cursor.fetchall())
    
    def save_model(self, name: str, model_type: str, file_path: str, 
                  accuracy: float = None, metadata: Dict[str, Any] = None) -> int:
        """
        Save model information to database
        
        Args:
            name (str): Model name
            model_type (str): Type of model
            file_path (str): Path to model file
            accuracy (float): Model accuracy
            metadata (dict): Additional metadata
            
        Returns:
            int: Model ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute('''
                INSERT INTO models (name, model_type, file_path, accuracy, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, model_type, file_path, accuracy, metadata_json))
            
            model_id = cursor.lastrowid
            conn.commit()
            
            return model_id
    
    def get_models(self) -> List[Tuple[int, str, str, str, float, Dict[str, Any]]]:
        """
        Get all saved models
        
        Returns:
            list: List of (id, name, model_type, file_path, accuracy, metadata) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, model_type, file_path, accuracy, metadata
                FROM models ORDER BY created_at DESC
            ''')
            
            rows = cursor.fetchall()
            
            models = []
            for row in rows:
                model_id, name, model_type, file_path, accuracy, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                models.append((model_id, name, model_type, file_path, accuracy, metadata))
            
            return models
    
    def get_latest_model(self) -> Optional[Tuple[int, str, str, str, float, Dict[str, Any]]]:
        """
        Get the most recent model
        
        Returns:
            tuple: Model information or None
        """
        models = self.get_models()
        return models[0] if models else None
    
    def save_training_session(self, model_id: int, num_samples: int, accuracy: float,
                            duration: float, metadata: Dict[str, Any] = None) -> int:
        """
        Save training session information
        
        Args:
            model_id (int): Associated model ID
            num_samples (int): Number of samples used
            accuracy (float): Training accuracy
            duration (float): Training duration in seconds
            metadata (dict): Additional metadata
            
        Returns:
            int: Session ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute('''
                INSERT INTO training_sessions (model_id, num_samples, accuracy, duration, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_id, num_samples, accuracy, duration, metadata_json))
            
            session_id = cursor.lastrowid
            conn.commit()
            
            return session_id
    
    def get_training_sessions(self, model_id: int = None) -> List[Tuple[int, int, int, float, float, Dict[str, Any]]]:
        """
        Get training sessions
        
        Args:
            model_id (int): Filter by model ID (optional)
            
        Returns:
            list: List of training session information
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if model_id:
                cursor.execute('''
                    SELECT id, model_id, num_samples, accuracy, duration, metadata
                    FROM training_sessions WHERE model_id = ? ORDER BY created_at DESC
                ''', (model_id,))
            else:
                cursor.execute('''
                    SELECT id, model_id, num_samples, accuracy, duration, metadata
                    FROM training_sessions ORDER BY created_at DESC
                ''')
            
            rows = cursor.fetchall()
            
            sessions = []
            for row in rows:
                session_id, model_id, num_samples, accuracy, duration, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                sessions.append((session_id, model_id, num_samples, accuracy, duration, metadata))
            
            return sessions
    
    def save_setting(self, key: str, value: Any):
        """
        Save a setting
        
        Args:
            key (str): Setting key
            value: Setting value (will be JSON serialized)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            value_json = json.dumps(value)
            
            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value_json))
            
            conn.commit()
    
    def get_setting(self, key: str, default=None):
        """
        Get a setting
        
        Args:
            key (str): Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
            row = cursor.fetchone()
            
            if row:
                return json.loads(row[0])
            return default
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            dict: Database statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sample statistics
            cursor.execute('SELECT COUNT(*) FROM samples')
            total_samples = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT label) FROM samples')
            unique_labels = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT user_id) FROM samples')
            unique_users = cursor.fetchone()[0]
            
            # Model statistics
            cursor.execute('SELECT COUNT(*) FROM models')
            total_models = cursor.fetchone()[0]
            
            # Training session statistics
            cursor.execute('SELECT COUNT(*) FROM training_sessions')
            total_sessions = cursor.fetchone()[0]
            
            return {
                'total_samples': total_samples,
                'unique_labels': unique_labels,
                'unique_users': unique_users,
                'total_models': total_models,
                'total_training_sessions': total_sessions,
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024)
            }
    
    def backup_database(self, backup_path: str):
        """
        Create a backup of the database
        
        Args:
            backup_path (str): Path for backup file
        """
        import shutil
        shutil.copy2(self.db_path, backup_path)
    
    def export_samples_to_csv(self, csv_path: str, user_id: str = None):
        """
        Export samples to CSV file (features will be flattened)
        
        Args:
            csv_path (str): Path for CSV file
            user_id (str): Export samples for specific user (optional)
        """
        import pandas as pd
        
        samples = self.get_all_samples(user_id)
        
        if not samples:
            print("No samples to export")
            return
        
        # Prepare data for CSV
        data = []
        for sample_id, features, label in samples:
            row = {'id': sample_id, 'label': label}
            # Flatten features
            for i, feature in enumerate(features):
                row[f'feature_{i}'] = feature
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Exported {len(samples)} samples to {csv_path}") 