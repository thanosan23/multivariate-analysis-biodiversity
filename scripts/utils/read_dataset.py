import sqlite3
import pandas as pd


def read_dataset(data={}, aggregate_col='Year'):
    query = "SELECT "
    cols = []
    first_key = (list(data.keys()))[0]
    query += f"{first_key}.\"{aggregate_col}\", "
    total = 0
    for key, value in data.items():
        total += 1
        for col in value:
            if col != aggregate_col:
                query += f"{key}.\"{col}\", "
        cols.append(key)
    query = query[:-2] + ' ' # remove last comma
    query += "FROM "
    for col in cols:
        query += f"{col}, "
    query = query[:-2] + ' ' # remove last comma
    if total > 1:
        query += "WHERE "
        for i in range(len(cols) - 1):
            query += f"{cols[i]}.\"{aggregate_col}\" == {cols[i + 1]}.\"{aggregate_col}\" "
            if i != len(cols)-2:
                query += "AND "
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query(query, conn)
    conn.close()

    # first row is the headers
    df = df.iloc[1:]
    df.reset_index(drop=True, inplace=True)

    return df
