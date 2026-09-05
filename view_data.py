import sqlite3
import datetime

def main():
    print("=== RECENT BOT SIGNALS ===\n")
    try:
        conn = sqlite3.connect("data/bot.db")
        cur = conn.cursor()
        
        # Fetch the last 10 signals sent by the bot
        cur.execute("""
            SELECT ts, symbol, direction, label, score, status 
            FROM signals 
            ORDER BY ts DESC 
            LIMIT 10
        """)
        rows = cur.fetchall()
        
        if not rows:
            print("No signals found in the database yet.")
            return

        print(f"{'Date & Time':<22} | {'Symbol':<8} | {'Score':<6} | {'Label':<15} | {'Status'}")
        print("-" * 75)
        
        for row in rows:
            ts, symbol, direction, label, score, status = row
            # Convert Unix timestamp (milliseconds) to readable date
            dt = datetime.datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{dt:<22} | {symbol:<8} | {score:<+6.1f} | {label:<15} | {status}")
            
    except Exception as e:
        print(f"Error reading database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
