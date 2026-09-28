import pymysql

connection = pymysql.connect(
    host='localhost',
    user='chestvision_user',
    password='ChestVision@123',
    database='chestvision'
)

with connection.cursor() as cursor:
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN status VARCHAR(50) NOT NULL DEFAULT 'pending'")
        print("Success: added status column.")
    except Exception as e:
        print(f"Error (maybe column exists): {e}")

connection.commit()
connection.close()
