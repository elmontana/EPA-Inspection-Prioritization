from datetime import datetime
import parsedatetime as pdt



def parse_date(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')


def parse_interval(time_str):
    cal = pdt.Calendar()
    return cal.parseDT(time_str, sourceTime=datetime.min)[0] - datetime.min


def date_to_string(date):
    return date.strftime('%Y-%m-%d')


def get_current_time_string():
    now = datetime.now()
    return now.strftime('%Y%m%d%H%M%S')
