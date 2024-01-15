import pandas
import pyodbc

from data.learning_model import LearningModel

class DataBase:
    def __init__(self):
        database_con = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                                      'Server=(localdb)\MSSQLLocalDB;'
                                      'Database=Load_Database;'
                                      'Trusted_Connection=yes;')
        cursor = database_con.cursor()
        # cursor.execute('DROP TABLE dbo.ModelLearningData')
        # cursor.execute('DROP TABLE dbo.AverageLoad')
        # cursor.execute(
	    #                     'CREATE TABLE dbo.ModelLearningData ('
        #                         'Year smallint not null,'
		#                         'Month tinyint not null,'
        #                         'Day tinyint not null,'
        #                         'Hour tinyint not null,'
		#                         'Temp real not null,'
		#                         'Feelslike real not null,'
		#                         'Humidity real not null,' 
		#                         'WindSpeed real not null,'
        #                         'CloudCover real not null,'
        #                         'WeeakDay tinyint not null,' 
        #                         'Daylight bit not null,'
		#                         'Load real not null'
	    #                     ')')
        # cursor.execute(
	    #                     'CREATE TABLE dbo.AverageLoad ('
        #                         'Year smallint not null,'
		#                         'Month tinyint not null,'
        #                         'Day tinyint not null,'
		#                         'AvgLoad real not null'
	    #                     ')')
        # cursor.execute('DELETE FROM dbo.ModelLearningData WHERE 1=1')
        # cursor.execute('DELETE FROM dbo.AverageLoad WHERE 1=1')
        # #cursor.execute('DELETE FROM dbo.ModelLearningDataInput WHERE 1=1')
        # #cursor.execute('DELETE FROM dbo.AverageLoadInput WHERE 1=1')
        # #cursor.execute('SELECT TOP(5) * FROM dbo.ModelLearningData')
        # cursor.execute('SELECT TOP(5) * FROM dbo.AverageLoad')
        # #cursor.execute('SELECT TOP(5) * FROM dbo.ModelLearningDataInput')
        # for row in cursor:
        #    print(row)
        #cursor.execute('SELECT TOP(5) * FROM dbo.AverageLoadInput')
        # for row in cursor:
        #    print(row)
        database_con.commit()
        cursor.close()
        database_con.close()

    def connect(self):
        return pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                                      'Server=(localdb)\MSSQLLocalDB;'
                                      'Database=Load_Database;'
                                      'Trusted_Connection=yes;')
        
    def clean_data_input(self):
        database_conn = self.connect()
        cursor = database_conn.cursor()
        
        cursor.execute('DELETE FROM dbo.ModelLearningDataInput WHERE 1=1')
        
        database_conn.commit()
        cursor.close()
        database_conn.close()

    def add_element(self, element: LearningModel, learning: bool):
        database_con = self.connect()
        cursor = database_con.cursor()

        input = ''
        if learning == False:
            input = 'Input'

        sql = 'INSERT INTO dbo.ModelLearningData'+ input +' (Year, Month, Day, Hour, Temp, Feelslike, Humidity, WindSpeed, CloudCover, WeeakDay, Daylight, Load) ' \
              'VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(
                element.year, element.month, element.day, element.hour, element.temp, element.feels_like, element.humidity, element.wind_speed, element.cloud_cover, element.week_day, element.daylight, element.load
              )
        cursor.execute(sql)
        database_con.commit()
        cursor.close()
        database_con.close()

    def add_average_load(self, year, month, day, avg_load, learning):
        database_con = self.connect()
        cursor = database_con.cursor()

        input = ''
        if learning == False:
            input = 'Input'

        sql = 'INSERT INTO dbo.AverageLoad' + input + ' (Year, Month, Day, AvgLoad) ' \
              'VALUES ({}, {}, {}, {})'.format(
                year, month, day, avg_load
              )
        cursor.execute(sql)
        database_con.commit()
        cursor.close()
        database_con.close()

    def get_pandas_dataframe(self, yearFrom, monthFrom, dayFrom, yearTo, monthTo, dayTo, learningData):
        database_con = self.connect()
        cursor = database_con.cursor()
        #rows = cursor.execute(f'SELECT * FROM dbo.ModelLearningData where Month BETWEEN {monthFrom} AND {monthTo} AND Year BETWEEN {yearFrom} AND {yearTo} AND Day BETWEEN {dayFrom} AND {dayTo}')
        
        # Example input dates (year, month, day)
        date_from = (dayFrom, monthFrom, yearFrom)  # 1st January 2023
        date_to = (dayTo, monthTo, yearTo)   # 31st January 2023

        # Converting to 'YYYY-MM-DD' format
        date_from_str = f"{date_from[2]}-{str(date_from[1]).zfill(2)}-{str(date_from[0]).zfill(2)}"
        date_to_str = f"{date_to[2]}-{str(date_to[1]).zfill(2)}-{str(date_to[0]).zfill(2)}"

        input = ""
        if learningData is False:
            input = "Input"        
        
        # Your SQL query
        query = f"""
        SELECT *
        FROM dbo.ModelLearningData{input}
        WHERE CAST(
                CAST([Year] AS VARCHAR(4)) + '-' + 
                RIGHT('0' + CAST([Month] AS VARCHAR(2)), 2) + '-' + 
                RIGHT('0' + CAST([Day] AS VARCHAR(2)), 2)
            AS DATETIME) 
            BETWEEN '{date_from_str}' AND '{date_to_str}'
        """
        
        # if learningData == True:  
        #     sql = 'SELECT * FROM dbo.GetScaledModelVer2({},{},{},{},{},{})'.format(
        #         yearFrom, monthFrom, dayFrom, yearTo, monthTo, dayTo
        #     )
        # else:
        #     sql = 'SELECT * FROM dbo.GetScaledModelTestData({},{},{},{},{},{})'.format(
        #         yearFrom, monthFrom, dayFrom, yearTo, monthTo, dayTo
        #     )
        # rows = cursor.execute(sql)
        df = pandas.read_sql(query, database_con) 
        cursor.close()
        database_con.close()
        return df

    def get_max_load(self):
        database_con = self.connect()
        cursor = database_con.cursor()

        sql = 'SELECT dbo.GetMaxLoad()'
        rows = cursor.execute(sql)
        for val in rows:
            ret = val

        cursor.close()
        database_con.close()
        return ret[0]

    def get_min_load(self):
        database_con = self.connect()
        cursor = database_con.cursor()

        sql = 'SELECT dbo.GetMinLoad()'
        rows = cursor.execute(sql)
        for val in rows:
            ret = val

        cursor.close()
        database_con.close()
        return ret[0]