import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect("ecommerce.db")
cur = conn.cursor()
cur.execute("""
            CREATE TABLE IF NOT EXISTS orders(
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                amount REAL,
                order_date TEXT
            )""")

cur.execute("INSERT INTO order (user_id, amount, order_date) VALUES (?, ?, ?)",(1, 19.9, "2025-11-16"))

conn.commit()
cur.execute("SELECT user_id, SUM(amount) AS total FROM order GROUP BY user_id")
for row in cur.fetchall():
    print(row)

conn.close()

