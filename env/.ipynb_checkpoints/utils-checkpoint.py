# 时间计算函数
def time_to_seconds(time):
    hour = time // 10000000
    minute = (time // 100000) % 100
    second = (time // 1000) % 100
    millisecond = time % 1000
    total_seconds = hour * 3600 + minute * 60 + second + millisecond / 1000.0
    return total_seconds

def seconds_to_time(seconds):
    hour = int(seconds // 3600)
    minute = int((seconds % 3600) // 60)
    second = int(seconds % 60)
    millisecond = int((seconds * 1000) % 1000)
    time = hour * 10000000 + minute * 100000 + second * 1000 + millisecond
    return time

def time_delete(time_1, time_2):
    '''
    input: time_1, time2(format: HHMMSSsss; 93000000-> 9:30:00 000)
    output: time1- time2 (format: HHMMSSsss)
    '''
    time_1_sec, time_2_sec = time_to_seconds(time_1), time_to_seconds(time_2)
    delta_time_sec = time_1_sec - time_2_sec
    delta_time = seconds_to_time(delta_time_sec)
    return delta_time 