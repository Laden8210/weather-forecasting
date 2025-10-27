#!/usr/bin/env python3
"""
Database initialization script for AI prediction logs
Run this script to initialize the SQLite database
"""

import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database import db_manager

def main():
    """Initialize the database"""
    print("Initializing AI prediction logs database...")
    
    try:
        # The database is automatically initialized when DatabaseManager is created
        print("✅ Database initialized successfully!")
        print(f"📁 Database file: {db_manager.db_path}")
        
        # Test database connection
        stats = db_manager.get_prediction_statistics()
        print("✅ Database connection test successful!")
        print(f"📊 Current statistics: {stats}")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
