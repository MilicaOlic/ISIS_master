import os
import zoneinfo
from numpy import NaN
import pytz
import pandas as pd
from datetime import datetime, date
from astral.sun import sun
from astral import LocationInfo
import math
from data.learning_model import LearningModel
from data.database import DataBase

class DataPreprocessor:
    def __init__(self, files_folder_path):
        self.files_folder_path = files_folder_path
        self.town = 'N.Y.C.'
        self.database = DataBase()

    def load_from_fs(self):
        loads = {}
        temps = []
        lastNonCompleteModel: LearningModel
        nonCompleteExists = False
        one_day_load = []
        current_day = None
        missing_datetime = None

        print('Loading of load data starts at ', datetime.now())
        # C:\Users\Centar\Desktop\66f44a_f15540d80467442ea1e768f32fbdf5e2\Training Data\NYS Load  Data
        for id, dirs in enumerate(os.walk("C:\\Users\\Centar\\Desktop\\66f44a_f15540d80467442ea1e768f32fbdf5e2\\Training Data\\NYS Load  Data")):
            if(id == 0):
                continue
            for file in dirs[2]:
                load_data = pd.read_csv(os.path.join(dirs[0], file))
                for i in range(len(load_data.index)):
                    load_object = load_data.iloc[i]
                    if ((load_object['Time Stamp'])[-5:]).strip() == '00:00' and load_object['Name'].strip() == self.town:
                        if str(load_object['Load']) != 'nan':
                            loads[datetime.strptime(load_object['Time Stamp'], '%m/%d/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S').replace(' ', 'T')] =  load_object['Load'] 
                        else:
                            missing_datetime = load_object['Time Stamp']
                            continue
                      
                        if current_day == None:
                            current_day = datetime.strptime(load_object['Time Stamp'], '%m/%d/%Y %H:%M:%S').strftime('%Y-%m-%d')
                        else:
                            if current_day == datetime.strptime(load_object['Time Stamp'], '%m/%d/%Y %H:%M:%S').strftime('%Y-%m-%d'):
                                one_day_load.append(load_object['Load'])
                            else:
                                avg = sum(one_day_load) / len(one_day_load)
                                self.database.add_average_load(int(current_day[0:4]), int(current_day[5:7]), int(current_day[8:10]), avg, True)
                                one_day_load.clear()
                                one_day_load.append(load_object['Load'])
                                current_day = datetime.strptime(load_object['Time Stamp'], '%m/%d/%Y %H:%M:%S').strftime('%Y-%m-%d')
                    elif missing_datetime != None: 
                        print(missing_datetime)
                        if load_object['Name'].strip() == self.town and str(load_object['Load']) != 'nan':
                            loads[datetime.strptime(load_object['Time Stamp'], '%m/%d/%Y %H:%M:%S').strftime('%Y-%m-%d %H:00:00').replace(' ', 'T')] =  load_object['Load']
                            missing_datetime = None
                            if current_day == datetime.strptime(load_object['Time Stamp'], '%m/%d/%Y %H:%M:%S').strftime('%Y-%m-%d'):
                                one_day_load.append(load_object['Load'])
                            else:
                                avg = sum(one_day_load) / len(one_day_load)
                                self.database.add_average_load(int(current_day[0:4]), int(current_day[5:7]), int(current_day[8:10]), avg, True)
                                one_day_load.clear()
                                one_day_load.append(load_object['Load'])
                                current_day = datetime.strptime(load_object['Time Stamp'], '%m/%d/%Y %H:%M:%S').strftime('%Y-%m-%d')
                      
        print('Loading of load data ends at {}', datetime.now())       
            
        for id, dirs in enumerate(os.walk("C:\\Users\\Centar\\Desktop\\66f44a_f15540d80467442ea1e768f32fbdf5e2\\Training Data\\NYS Weather Data\\New York City, NY")):
            for file in dirs[2]:
                weather_data = pd.read_csv(os.path.join(dirs[0], file))
                for i in range(len(weather_data.index)):
                    weather_object = weather_data.iloc[i]
                    #loads_date = datetime.strptime(weather_object['datetime'], '%Y-%m-%dT%H:%M:%S').replace(' ', 'T')

                    load_object_value = loads.get(weather_object['datetime'])
                    if str(load_object_value) == 'nan' or str(load_object_value) == 'None':
                        print('ERROR: No load for {}', weather_object['datetime'])
                        continue

                    if str(weather_object['humidity']) == 'nan':
                        continue

                    if weather_object['humidity'] < 0 or weather_object['humidity'] > 100:
                        continue

                    if str(weather_object['temp']) == 'nan' or weather_object['temp'] < -20 or weather_object['temp'] > 115:
                        if nonCompleteExists:
                            print('ERROR: More than 1 temp missing in a row date-{}', weather_data['datetime'])
                            exit()
                        nonCompleteExists = True
                        lastNonCompleteModel = LearningModel(int(weather_object['datetime'][0:4]), int(weather_object['datetime'][5:7]), int(weather_object['datetime'][8:10]), int(weather_object['datetime'][11:13]), 0, weather_object['feelslike'],
                        weather_object['humidity'], weather_object['windspeed'], weather_object['cloudcover'], #weather_object['solarradiation'],
                        date(int(weather_object['datetime'][0:4]), int(weather_object['datetime'][5:7]), int(weather_object['datetime'][8:10])).weekday(), self.calculate_daylight(weather_object['datetime']), load_object_value)
                        continue

                    if nonCompleteExists:
                        lastNonCompleteModel.temp = (temps[-1] + weather_object['temp']) / 2
                        lastNonCompleteModel.feels_like = self.calculate_feelslike_temp(lastNonCompleteModel.temp, lastNonCompleteModel.wind_speed, lastNonCompleteModel.humidity)
                        self.write_to_db(lastNonCompleteModel, True)
                        nonCompleteExists = False
                    
                    temps.append(weather_object['temp'])
                    self.write_to_db(LearningModel(int(weather_object['datetime'][0:4]), int(weather_object['datetime'][5:7]), int(weather_object['datetime'][8:10]), int(weather_object['datetime'][11:13]), weather_object['temp'], weather_object['feelslike'],
                        weather_object['humidity'], weather_object['windspeed'], weather_object['cloudcover'], #weather_object['solarradiation'],
                        date(int(weather_object['datetime'][0:4]), int(weather_object['datetime'][5:7]), int(weather_object['datetime'][8:10])).weekday(), self.calculate_daylight(weather_object['datetime']), load_object_value), True)


        print('Loading of weather data ends at {}', datetime.now())         


    def write_to_db(self, learningModel: LearningModel, learning: bool):
        if str(learningModel.wind_speed) == 'nan':
            learningModel.wind_speed = 0
        if str(learningModel.feels_like) == 'nan':
            learningModel.feels_like = self.calculate_feelslike_temp(learningModel.temp, learningModel.wind_speed, learningModel.humidity)
        if str(learningModel.cloud_cover) == 'nan':
            learningModel.cloud_cover = 0
        self.database.add_element(learningModel, learning)
        

    def calculate_daylight(self, datetimeLight):
        city = LocationInfo("New York","New York","America/New_York", 40.7527, -73.9772)
        tz = zoneinfo.ZoneInfo("America/New_York")
        s = sun(city.observer, date=date(int(datetimeLight[0:4]),int(datetimeLight[5:7]),int(datetimeLight[8:10])), tzinfo=tz)
        est = pytz.timezone('US/Eastern')

        if est.localize(datetime.strptime(datetimeLight, '%Y-%m-%dT%H:%M:%S')) >= s['sunrise'] and \
           est.localize(datetime.strptime(datetimeLight, '%Y-%m-%dT%H:%M:%S')) <= s['sunset']:
            return 1
        else:
            return 0

          
    def calculate_feelslike_temp(self, vTemperature, vWindSpeed, vRelativeHumidity):
        if vTemperature <= 50 and vWindSpeed >= 3:
            vFeelsLike = 35.74 + (0.6215*vTemperature) - 35.75*(vWindSpeed**0.16) + ((0.4275*vTemperature)*(vWindSpeed**0.16))
        else:
            vFeelsLike = vTemperature
 
        # Replace it with the Heat Index, if necessary
        if vFeelsLike == vTemperature and vTemperature >= 80:
            vFeelsLike = 0.5 * (vTemperature + 61.0 + ((vTemperature-68.0)*1.2) + (vRelativeHumidity*0.094))
        
        if vFeelsLike >= 80:
            vFeelsLike = -42.379 + 2.04901523*vTemperature + 10.14333127*vRelativeHumidity - .22475541*vTemperature*vRelativeHumidity - .00683783*vTemperature*vTemperature - .05481717*vRelativeHumidity*vRelativeHumidity + .00122874*vTemperature*vTemperature*vRelativeHumidity + .00085282*vTemperature*vRelativeHumidity*vRelativeHumidity - .00000199*vTemperature*vTemperature*vRelativeHumidity*vRelativeHumidity
        if vRelativeHumidity < 13 and vTemperature >= 80 and vTemperature <= 112:
            vFeelsLike = vFeelsLike - ((13-vRelativeHumidity)/4)*math.sqrt((17-math.fabs(vTemperature-95.))/17)
        if vRelativeHumidity > 85 and vTemperature >= 80 and vTemperature <= 87:
                vFeelsLike = vFeelsLike + ((vRelativeHumidity-85)/10) * ((87-vTemperature)/5)

        return vFeelsLike

    def load_uploaded_files(self):
        self.database.clean_data_input()
        path = self.getLocalFolder()
        path.append('uploaded_files')
        path = '\\'.join(path)

        loads = {}
        weathers = []
        
        for files in enumerate(os.walk(path)):
            temps = []
            lastNonCompleteModel: LearningModel
            nonCompleteExists = False
            
            for file in (files[1])[2]:
                data = pd.read_csv(os.path.join(path, file))
                
                for i in range(len(data.index)):
                    weather_object = data.iloc[i]

                    if str(weather_object['humidity']) == 'nan':
                        continue

                    if weather_object['humidity'] < 0 or weather_object['humidity'] > 100:
                        continue

                    if str(weather_object['temp']) == 'nan' or weather_object['temp'] < -20 or weather_object['temp'] > 115:
                        if nonCompleteExists:
                            print('ERROR: More than 1 temp missing in a row date-{}', data['datetime'])
                            exit()
                        nonCompleteExists = True
                        lastNonCompleteModel = LearningModel(int(weather_object['datetime'][0:4]), int(weather_object['datetime'][5:7]), int(weather_object['datetime'][8:10]), int(weather_object['datetime'][11:13]), 0, weather_object['feelslike'],
                        weather_object['humidity'], weather_object['windspeed'], weather_object['cloudcover'], #weather_object['solarradiation'],
                        date(int(weather_object['datetime'][0:4]), int(weather_object['datetime'][5:7]), int(weather_object['datetime'][8:10])).weekday(), self.calculate_daylight(weather_object['datetime']), NaN)
                        continue

                    if nonCompleteExists:
                        lastNonCompleteModel.temp = (temps[-1] + weather_object['temp']) / 2
                        lastNonCompleteModel.feels_like = self.calculate_feelslike_temp(lastNonCompleteModel.temp, lastNonCompleteModel.wind_speed, lastNonCompleteModel.humidity)
                        weathers.append(lastNonCompleteModel)
                        nonCompleteExists = False
                    
                    temps.append(weather_object['temp'])
                    weathers.append(LearningModel(int(weather_object['datetime'][0:4]), int(weather_object['datetime'][5:7]), int(weather_object['datetime'][8:10]), int(weather_object['datetime'][11:13]), weather_object['temp'], weather_object['feelslike'],
                        weather_object['humidity'], weather_object['windspeed'], weather_object['cloudcover'], #weather_object['solarradiation'],
                        date(int(weather_object['datetime'][0:4]), int(weather_object['datetime'][5:7]), int(weather_object['datetime'][8:10])).weekday(), self.calculate_daylight(weather_object['datetime']), NaN))
            
            break

        for model in weathers:
            model.load = 0
            self.write_to_db(model, False)

    def getLocalFolder(self):
        path=str(os.path.dirname(os.path.abspath(__file__))).split(os.sep)
        return path[:-1]

    def process_data(self):
        self.load_uploaded_files()
        #self.load_from_fs()

    def get_all_models(self):
        for id, dirs in enumerate(os.walk(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))):
            return dirs[2]

    