import sqlite3
import argparse
import json

def list_entries(db_path='backtest_data.db', limit=5):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"Database: {db_path}")
    print("=" * 80)
    
    for table in tables:
        if table == 'sqlite_sequence':
            continue
            
        print(f"\nTable: {table}")
        print("-" * 80)
        
        # Get count
        cursor.execute(f"SELECT count(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"Total entries: {count}")
        
        if count > 0:
            print(f"\nFirst {limit} entries:")
            # Get column names
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Format header
            header = " | ".join(f"{col[:15]:<15}" for col in columns)
            print(header)
            print("-" * len(header))
            
            # Get data
            cursor.execute(f"SELECT * FROM {table} LIMIT {limit};")
            rows = cursor.fetchall()
            
            for row in rows:
                formatted_row = []
                for item in row:
                    item_str = str(item)
                    if len(item_str) > 15:
                        item_str = item_str[:12] + "..."
                    formatted_row.append(f"{item_str:<15}")
                print(" | ".join(formatted_row))
        print("=" * 80)
        
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List entries in SQLite database.')
    parser.add_argument('--db', type=str, default='backtest_data.db', help='Path to the SQLite database')
    parser.add_argument('--limit', type=int, default=5, help='Number of rows to display per table')
    args = parser.parse_args()
    
    list_entries(args.db, args.limit)
