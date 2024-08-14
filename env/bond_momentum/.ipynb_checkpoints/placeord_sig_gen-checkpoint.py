import multiprocessing
import sys
sys.path.append("../")
from env.utils import time_delete, time_to_seconds
import pandas as pd
import numpy as np
class PlaceOrdSigGen(object):
    def __init__(
        self,
        time_window: int
    ):
        self.time_window: int = 3 * time_window
        print('time_window:', time_window)
        self.lock = multiprocessing.Lock()
        self.data_cache: list[dict] = []
        self.vwap_data_cache: list[dict] = []
        
        self.debug_log_o = open("./signal_debuglog.csv", 'w')
        log = "time,askprice_0,bidprice_0,cache_len,vwap,cache_size,ask_bid_spread,ab_volume_misbalance,transaction_net_volume,volatility\n"
        self.debug_log_o.write(log)
        
        self.open_price = None 
        self.market_feature_num = 5
        
    def add_data(
        self,
        snap_shot: dict
    ) -> tuple():
        # 使用open_price, parent order quantity 作用归一化基准
        if not self.open_price:
            self.open_price = snap_shot['Open']
        
        with self.lock:
            current_time = snap_shot['Time']
            turnover = snap_shot['Turnover']
            volume = snap_shot['Volume']
            
            self.data_cache.append(snap_shot) 
            # 清理超出时间窗口的数据
            self.__clean_old_data(current_time)

            # 计算feature sequence 
            vwap, ask_bid_spread, ab_volume_misbalance, transaction_net_volume, volatility = self.__calculate_features()

            # 记录日志
            self.__log_debug(snap_shot, vwap, ask_bid_spread, ab_volume_misbalance, transaction_net_volume, volatility)
            return [vwap, ask_bid_spread, ab_volume_misbalance, transaction_net_volume, volatility]
    def add_vwap_data(
        self,
        snap_shot: dict):
        with self.lock:
            self.vwap_data_cache.append(snap_shot) 

    def __calculate_tot_vwap(self):
        # 计算 VWAP
        total_volume = self.vwap_data_cache[-1]['Volume'] - self.vwap_data_cache[0]['Volume']
        if total_volume == 0:
            vwap = 0.0
        else:
            vwap = (self.vwap_data_cache[-1]['Turnover'] - self.vwap_data_cache[0]['Turnover']) / total_volume * 1e4 # 1e4 是因为数据平台对原始price 乘1e4
        return vwap
   
    def get_tot_vwap(self):
        return self.__calculate_tot_vwap()
        
    def get_data(self) -> list[dict]:
        return self.data_cache
        
    def __clean_old_data(
        self,
        current_time: int
    ):
        while self.data_cache and time_delete(current_time, self.data_cache[0]['Time']) >= self.time_window:
            # print(time_delete(current_time, self.data_cache[0]['Time']), current_time, self.data_cache[0]['Time'])
            self.data_cache.pop(0)
            
    def __calculate_features(self) -> tuple:
        # 计算 VWAP
        total_volume = self.data_cache[-1]['Volume'] - self.data_cache[0]['Volume']
        if total_volume == 0:
            vwap = (self.data_cache[-1]['BidPrice'][0] + self.data_cache[-1]['AskPrice'][0]) / 2
        else:
            vwap = (self.data_cache[-1]['Turnover'] - self.data_cache[0]['Turnover']) / total_volume * 1e4 # 1e4 是因为数据平台对原始price 乘1e4
        
        # 计算统计量因子
        bid_prices = [data['BidPrice'][0] for data in self.data_cache]
        ask_prices = [data['AskPrice'][0] for data in self.data_cache]
        bid_volumes = [sum(data['BidVolume']) for data in self.data_cache]
        ask_volumes = [sum(data['AskVolume']) for data in self.data_cache]
        
        # 买卖价差
        ask_bid_spread = np.mean(ask_prices) - np.mean(bid_prices)
        
        # 买卖量不平衡
        ab_volume_misbalance = np.sum(bid_volumes) - np.sum(ask_volumes)
        
        # 总成交股数
        transaction_net_volume = total_volume
        
        # 波动率
        mid_prices = [(data['AskPrice'][0] + data['BidPrice'][0]) / 2 for data in self.data_cache]
        volatility = np.std(mid_prices)
        
        return vwap, ask_bid_spread, ab_volume_misbalance, transaction_net_volume, volatility
    
    def __log_debug(self, snap_shot: dict, vwap: float, ask_bid_spread: float, ab_volume_misbalance: float, transaction_net_volume: float, volatility: float):
        log = f"{snap_shot['Time']},{snap_shot['AskPrice'][0]},{snap_shot['BidPrice'][0]},{len(self.data_cache)}, {vwap},{len(self.data_cache)},{ask_bid_spread},{ab_volume_misbalance},{transaction_net_volume},{volatility}\n"
        self.debug_log_o.write(log)
        self.debug_log_o.flush()