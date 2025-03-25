import sqlite3

class Database:
    def __init__(self):
        self.execute(
            '''
            CREATE TABLE IF NOT EXISTS credentials (
                uid INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(25) NOT NULL,
                password VARCHAR(25) NOT NULL
            );
            '''
        )

    @staticmethod
    def _connect():
        db = sqlite3.connect("db/database.db")
        cur = db.cursor()
        return db, cur

    @staticmethod
    def _fetchone(cursor: sqlite3.Cursor):
        return cursor.fetchone()
    
    @staticmethod
    def _fetchall(cursor: sqlite3.Cursor):
        return cursor.fetchall()

    @staticmethod
    def _commit(db: sqlite3.Connection):
        cursor = db.cursor()
        db.commit()
        cursor.close()
        db.close()

    def execute(self, query: str, *values: tuple):
        db, cursor = self._connect()
        cursor.execute(query, values)
        self._commit(db)

    def fetchOne(self, query: str, *values: tuple):
        db, cursor = self._connect()
        cursor.execute(query, values)
        result = self._fetchone(cursor)
        db.close()
        return result

    def fetchAll(self, query: str, *values: tuple):
        db, cursor = self._connect()
        cursor.execute(query, values)
        result = self._fetchall(cursor)
        db.close()
        return result
    
    def fetchVal(self, query: str, *values: tuple):
        result = self.fetchOne(query, *values)
        return result[0]


