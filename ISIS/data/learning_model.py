class LearningModel():
    year = None
    month = None
    day = None
    hour = None
    temp = None 
    feels_like = None 
    humidity = None 
    wind_speed = None
    cloud_cover = None 
    week_day = None 
    daylight = None 
    load = None

    def __init__(self, year, month, day, hour, temp, feels_like, humidity, wind_speed,
                 cloud_cover, week_day, daylight, load):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.temp = temp 
        self.feels_like = feels_like 
        self.humidity = humidity 
        self.wind_speed = wind_speed
        self.cloud_cover = cloud_cover 
        self.week_day = week_day 
        self.daylight = daylight 
        self.load = load