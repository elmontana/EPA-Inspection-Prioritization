from datetime import datetime
from dateutil.relativedelta import relativedelta



def parse_date(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')


def parse_interval(time_str):
    if ' ' in time_str:
        num = int(time_str.split(' ')[0])
        unit = time_str.split(' ')[1][0].lower()
    else:
        num = int(time_str[:-1])
        unit = time_str[-1]
    if unit == 'y':
        return relativedelta(years=num)
    elif unit == 'm':
        return relativedelta(months=num)
    elif unit == 'd':
        return relativedelta(days=num)
    else:
        raise ValueError("Parse interval doesn't support unit {unit}.")

def date_to_string(date):
    return date.strftime('%Y-%m-%d')


def get_current_time_string():
    now = datetime.now()
    return now.strftime('%Y%m%d%H%M%S')
