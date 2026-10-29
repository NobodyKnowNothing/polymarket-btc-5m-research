import sqlite3

def cleanup():
    db_path = 'backtest_data.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Analyzing database...")
    
    # Get all unique slugs from snapshots
    cursor.execute("SELECT DISTINCT market_slug FROM market_snapshots")
    all_slugs = set(row[0] for row in cursor.fetchall())
    print(f"Total unique market slugs in snapshots: {len(all_slugs)}")

    # Get slugs that HAVE a conclusion
    cursor.execute("SELECT market_slug FROM market_outcomes WHERE actual_outcome IS NOT NULL AND actual_outcome != 'None'")
    concluded_slugs = set(row[0] for row in cursor.fetchall())
    print(f"Slugs with conclusions: {len(concluded_slugs)}")

    slugs_to_remove = all_slugs - concluded_slugs
    print(f"Slugs to remove (no conclusion): {len(slugs_to_remove)}")

    if not slugs_to_remove:
        print("No data to remove.")
        conn.close()
        return

    # Delete snapshots for slugs without conclusions
    print(f"Deleting snapshots for {len(slugs_to_remove)} slugs...")
    placeholders = ', '.join(['?'] * len(slugs_to_remove))
    cursor.execute(f"DELETE FROM market_snapshots WHERE market_slug NOT IN (SELECT market_slug FROM market_outcomes WHERE actual_outcome IS NOT NULL AND actual_outcome != 'None')")
    snapshots_deleted = cursor.rowcount
    print(f"Deleted {snapshots_deleted} rows from market_snapshots.")

    # Delete outcome entries that don't have a conclusion
    cursor.execute("DELETE FROM market_outcomes WHERE actual_outcome IS NULL OR actual_outcome = 'None'")
    outcomes_deleted = cursor.rowcount
    print(f"Deleted {outcomes_deleted} rows from market_outcomes.")

    # Commit the deletions
    conn.commit()

    # Optional: Vacuum the database to reclaim space
    print("Vacuuming database...")
    conn.isolation_level = None  # Disable transaction for vacuum
    conn.execute("VACUUM")
    conn.isolation_level = '' # Restore default
    
    conn.close()
    print("Cleanup complete.")

if __name__ == "__main__":
    cleanup()
