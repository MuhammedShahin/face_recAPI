import io
import sqlite3
from sqlite3 import Error
import numpy as np


DATABASE_PATH = 'database.db'

global list_of_students

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def select_form_table(conn , quary):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute(quary)

    rows = cur.fetchall()

    for row in rows:
        print(row)

def get_database():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        # print(sqlite3.version)
    except Error as e:
        print(e)

    return conn

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def insert_student(id, name, email, photo, section, year):
    try:
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)

        sqliteConnection = sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_insert_with_param = """INSERT INTO Students
                          (ID, Name, email, photo, section, year) 
                          VALUES (?, ?, ?, ?, ?, ?);"""
        # Converts np.array to TEXT when inserting
        print('type of the photo --------:', type(photo))
        data_tuple = (id, name, email, photo, section, year)
        cursor.execute(sqlite_insert_with_param, data_tuple)
        # sqliteConnection.commit()
        print("new student inserted!!")

        print("insert subjects for the student")
        sql_select_query = """select ID from Subjects where year = ?"""
        cursor.execute(sql_select_query, (year,))
        records = cursor.fetchall()
        for rows in records:
            sqlite_insert_with_param = """INSERT INTO Student_subject
                                     (Student_id, subject_id) 
                                     VALUES (?,?);"""
            # Converts np.array to TEXT when inserting

            data_tuple = (id, rows[0])
            cursor.execute(sqlite_insert_with_param, data_tuple)
        sqliteConnection.commit()
        for rows in records:
            sqlite_insert_with_param = """INSERT INTO Attendance
                                     (Student_id, subject_id) 
                                     VALUES (?, ?);"""
            # Converts np.array to TEXT when inserting

            data_tuple = (id, rows[0])
            cursor.execute(sqlite_insert_with_param, data_tuple)
            # sqliteConnection.commit()
        sqliteConnection.commit()
        print("Done!!")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert Python variable into sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")


def getstudentInfo(id):
    try:
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)

        sqliteConnection = sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = sqliteConnection.cursor()

        print("Connected to SQLite")

        sql_select_query = """select * from Students where ID = ?"""
        cursor.execute(sql_select_query, (id,))
        records = cursor.fetchall()
        print("Printing ID ", id)
        photo = None
        for row in records:
            print("Name = ", row[1])
            print("Email  = ", row[2])
            print("photo  = ", row[3])
            print("section  = ", row[4])
            photo = row[3]
        cursor.close()
        print(type(photo))
        print(photo)
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")


def create_student_table():
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)

    sqliteConnection = sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = sqliteConnection.cursor()

    create_table_sql = """CREATE TABLE "Students" (
	"ID"	INTEGER,
	"Name"	TEXT NOT NULL,
	"email"	TEXT NOT NULL UNIQUE,
	"photo"	array,
	"section"	INTEGER NOT NULL,
	"year" INTEGER,
	PRIMARY KEY("ID")
);"""

    cursor.execute(create_table_sql)
    cursor.close()
    sqliteConnection.close()


def get_sections_subject(TAID, subID):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    sql_select_query = """select id_TA_subject_section from TA_Subjects where ta_id = ? AND subjects_id = ?"""
    cursor.execute(sql_select_query, (TAID,subID))
    records = cursor.fetchall()
    RE_id = None
    if len(records) != 0:
        RE_id = records[0][0]

    sql_select_query1 = """select year from Subjects where ID = ?"""
    cursor.execute(sql_select_query1, (subID, ))
    records = cursor.fetchall()
    year = None
    if len(records) != 0:
        year = records[0][0]

    sql_select_query2 = """select section from RE_section_subject_TA where id_ta_subject = ?"""
    cursor.execute(sql_select_query2, (RE_id,))
    records = cursor.fetchall()
    lst_sections = [row[0] for row in records]
    conn.close()
    return lst_sections, year


# def get_sections(ID):
#     conn = sqlite3.connect(DATABASE_PATH)
#     cursor = conn.cursor()
#     sql_select_query = """select section from RE_section_subject_TA where id_ta_subject = ?"""
#     cursor.execute(sql_select_query, (ID,))
#     records = cursor.fetchall()
#     conn.close()
#     lst_sections = [row[0] for row in records]
#     if len(records) == 0:
#         return 'NO sections found'
#     return lst_sections


def login_TA(email , password):
    try:
        cheak_att =False
        sqliteConnection = sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = sqliteConnection.cursor()

        print("Connected to SQLite")

        sql_select_query = """select * from TA where email = ? and password= ?"""
        cursor.execute(sql_select_query, (email,password,))
        records = cursor.fetchall()
        if records==[]:
           print("Not valid email or password!!")
        else:
            print("ID = ", records[0][0])
            ID =  records[0][0]
            print("Name = ", records[0][1])
            Name = records[0][1]
            # print("Email  = ", row[2])
            print("Valid login for TA")
            cheak_att = True

        # featch all the subjects that tha TA is assigned in
        sql_select_query = """select subjects_id from TA_Subjects where ta_id = ?"""
        cursor.execute(sql_select_query, (ID,))
        records_of_subjects = cursor.fetchall()
        list_of_subjects = [row[0] for row in records_of_subjects]

        cursor.close()
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")
        if cheak_att:
            return ID , list_of_subjects , "Valid login"
        else:
            return None , None,"Not valid email or password!!"


def get_students_outof_section(section_number , year):
    try:
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)

        sqliteConnection = sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = sqliteConnection.cursor()

        print("Connected to SQLite")

        sql_select_query = """select ID , Name , photo from Students where section = ? and year= ?"""
        cursor.execute(sql_select_query, (section_number, year,))
        records = cursor.fetchall()
        list_of_students = [list(rows) for rows in records]
        # print(list_of_students)
        cursor.close()
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")
            return list_of_students


def clear_database():
    try:
        sqliteConnection = sqlite3.connect(DATABASE_PATH)
        cursor = sqliteConnection.cursor()

        print("Connected to SQLite")

        sql_select_query = """DELETE FROM RE_section_subject_ta"""
        sql_select_query1 = """DELETE FROM Attendance"""
        sql_select_query2 = """DELETE FROM TA_Subjects"""
        sql_select_query3 = """DELETE FROM Student_subject"""
        sql_select_query4 = """DELETE FROM Students"""
        sql_select_query5 = """DELETE FROM Subjects"""
        sql_select_query6 = """DELETE FROM TA"""

        # cursor.execute(sql_select_query)
        # sqliteConnection.commit()
        cursor.execute(sql_select_query1)
        sqliteConnection.commit()
        # cursor.execute(sql_select_query2)
        # sqliteConnection.commit()
        cursor.execute(sql_select_query3)
        sqliteConnection.commit()
        cursor.execute(sql_select_query4)
        sqliteConnection.commit()
        # cursor.execute(sql_select_query5)
        # sqliteConnection.commit()
        # cursor.execute(sql_select_query6)
        # sqliteConnection.commit()
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")
            return "DONE!"


def insert_ta(name, email, password ):
    try:

        sqliteConnection = sqlite3.connect(DATABASE_PATH)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_insert_with_param = """INSERT INTO TA
                          (Name, email, password) 
                          VALUES (?, ?, ?);"""
        # Converts np.array to TEXT when inserting
        # print('type of the photo --------:', type(photo))
        data_tuple = (name, email, password)
        cursor.execute(sqlite_insert_with_param, data_tuple)
        # sqliteConnection.commit()
        print("new student inserted!!")

        sqliteConnection.commit()
        print("Done!!")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert Python variable into sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")



def assign_subjectTO_ta(TA_ID, subject_id, lst_sections):
    try:

        sqliteConnection = sqlite3.connect(DATABASE_PATH)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_select = """select max(id_TA_subject_section) from TA_Subjects"""
        cursor.execute(sqlite_select)
        record = cursor.fetchall()
        id_TA_subject_section = None
        if record!=[(None,)]:
            id_TA_subject_section = record[0][0] + 1
        elif record==[(None,)]:
            id_TA_subject_section = 0
        else:
            id_TA_subject_section = 0

        sqlite_insert_with_param = """INSERT INTO TA_Subjects
                             (ta_id, subjects_id, id_TA_subject_section) 
                             VALUES (?, ?, ?);"""
        # Converts np.array to TEXT when inserting
        # print('type of the photo --------:', type(photo))
        data_tuple = (TA_ID, subject_id, id_TA_subject_section)
        cursor.execute(sqlite_insert_with_param, data_tuple)
        sqliteConnection.commit()


        for sec in lst_sections:
            sqlite_insert_with_param = """INSERT INTO RE_section_subject_TA
                                         (id_ta_subject, section) 
                                         VALUES (?, ?);"""
            # Converts np.array to TEXT when inserting
            # print('type of the photo --------:', type(photo))
            data_tuple = (id_TA_subject_section, sec)
            cursor.execute(sqlite_insert_with_param, data_tuple)
        sqliteConnection.commit()

        print("Done!!")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert Python variable into sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")


def insert_subject(subID, year):
    try:

        sqliteConnection = sqlite3.connect(DATABASE_PATH)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_insert_with_param = """INSERT INTO Subjects
                          (ID, year) 
                          VALUES (?, ?);"""

        data_tuple = (subID, year)
        cursor.execute(sqlite_insert_with_param, data_tuple)

        sqliteConnection.commit()
        print("Done!!")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert Python variable into sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")

def insert_attendance(student_id, subject_id , week_num):
    try:

        sqliteConnection = sqlite3.connect(DATABASE_PATH)
        cursor = sqliteConnection.cursor()
        week_str = "week" + str(week_num)
        print("Connected to SQLite")
        sqlite_insert_with_param = ''' UPDATE Attendance
              SET '''+week_str+''' = 1
              WHERE student_id = ? and subject_id = ?'''

        data_tuple = (student_id , subject_id)
        cursor.execute(sqlite_insert_with_param, data_tuple)

        sqliteConnection.commit()
        print("Done!!")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert Python variable into sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")

# def main_database():
#     # connenction = get_database()
#     # getstudentInfo(2016170334)
#     # create_student_table()
#     # select_stat = """select * from Subjects"""
#     # select_form_table(connenction, select_stat)
#     # print(get_sections(1))
#     print(get_students_outof_section(3 , 2))
#     koko=0
#
#
# main_database()
# insert_subject("Algorithms",3)
# insert_ta("Dina", "Dina@gmail.com","000000")
# assign_subjectTO_ta(1,"Architecture",[2,3,6,9,12,15])
# clear_database()
# insert_student(2016170341, "Gomaa", "Gomaa@gmail.com",np.array([1,2,3]), 3, 3)
# insert_attendance(2016170341, "AI" , 5)
# [2,3,6,9,12,15]
# [4,5,7,13,16,19]