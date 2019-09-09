import sqlite3

db = sqlite3.connect('beijing_bus.sqlite3')
print(db.execute('SELECT SQLITE_VERSION()').fetchall())
db.execute('')

db.close()
