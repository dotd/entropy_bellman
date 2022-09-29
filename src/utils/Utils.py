from datetime import datetime


def get_time_str():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")
