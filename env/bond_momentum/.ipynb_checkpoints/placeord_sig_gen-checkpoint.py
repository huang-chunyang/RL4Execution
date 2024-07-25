import multiprocessing
import sys
sys.path.append("../")
from utils import time_delete, time_to_seconds

class PlaceOrdSigGen(object):

    def __init__(
        self,
        time_window: int
    ):
        self.time_window: int = 3 * time_window
        print('time_window:', time_window)
        self.lock = multiprocessing.Lock()
        self.data_cache: list[dict] = []
        
        self.debug_log_o = open("./signal_debuglog.csv", 'w')
        log = "time,askprice_0,bidprice_0,cache_len,vwap,cache_size\n"
        self.debug_log_o.write(log)
        
    def add_data(
        self,
        snap_shot: dict
        # trade_detail: dict
    ) -> tuple():
        with self.lock:
            current_time = snap_shot['Time']
            turnover = snap_shot['Turnover']
            volume = snap_shot['Volume']
            
            self.data_cache.append(snap_shot) 
            # 清理超出时间窗口的数据
            self.__clean_old_data(current_time)

            # 计算VWAP
            vwap = self.__calculate_vwap()
            
            # 记录日志
            self.__log_debug(snap_shot, vwap)
            return vwap
    def get_data(self) -> list[dict]:
        return self.data_cache
        
    def __clean_old_data(
        self,
        current_time: int
    ):
        while self.data_cache and time_to_seconds(time_delete(current_time, self.data_cache[0]['Time'])) >= self.time_window:
            # print(time_delete(current_time, self.data_cache[0]['Time']), current_time, self.data_cache[0]['Time'])
            self.data_cache.pop(0)
            
    def __calculate_vwap(self) -> float:
        total_volume = self.data_cache[-1]['Volume'] - self.data_cache[0]['Volume']
        if total_volume == 0:
            return 0.0
        vwap = (self.data_cache[-1]['Turnover'] - self.data_cache[0]['Turnover']) / total_volume * 1e4 # 1e4 是因为数据平台对原始price 乘1e4
        return vwap
    
    def __log_debug(self, snap_shot: dict, vwap: float):
        log = f"{snap_shot['Time']},{snap_shot['AskPrice'][0]},{snap_shot['BidPrice'][0]},{len(self.data_cache)}, {vwap},{len(self.data_cache)}\n"
        self.debug_log_o.write(log)
        self.debug_log_o.flush()