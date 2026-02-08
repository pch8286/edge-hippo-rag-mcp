import sqlite3
try:
    conn = sqlite3.connect(":memory:")
    if hasattr(conn, "enable_load_extension"):
        print("enable_load_extension IS available")
    else:
        print("enable_load_extension IS NOT available")
except Exception as e:
    print(f"Error: {e}")
