#!/usr/bin/env python

# encoding: UTF-8

import os
import sys
sys.path.append("../")
from utils import time_to_seconds, seconds_to_time, time_delete
import multiprocessing
import threading
import queue

from simkelly_pyapi.kelly_pymd_api import KellyMdApi
from simkelly_pyapi.kelly_pytd_api import (KELLY_OT_LIMITPRICE, KELLY_D_BUY, KELLY_D_SELL,
                                           KELLY_MT_MATCH, KELLY_OS_ACCEPT, KELLY_OS_ALL_FILLED)

import config
from .common_def import TDSug
from .placeord_sig_gen import PlaceOrdSigGen

import matplotlib.pyplot as plt
import matplotlib.animation as animation

START_TIME = 91500000
START_TIME1 = 92000000
START_TIME2 = 92500000
AM_BEGIN = 93000000
AM_END = 113000000
PM_BEGIN = 130000000
PM_CLOSINGPOS_BEGIN = 143000000
PM_END1 = 145700000
PM_END = 150000000
OPEN_DURING = 4 * 60 * 60 * 1000.0
FREE_DURING = 1.5 * 60 * 60 * 1000.0

SH_SIDE_BUY = 'B'
SH_SIDE_SELL = 'S'
SZ_SIDE_BUY = '1'
SZ_SIDE_SELL = '2'

SH_ORDERKIND_ADD = 'A'
SH_ORDERKIND_DELETE = 'D'
SZ_ORDERKIND_MARKET_PRICE = '1'
SZ_ORDERKIND_FIXED_PRICE = '2'
SZ_ORDERKIND_OUR_BEST = 'U'

class TradeOrder:
    def __init__(self, start_time, end_time, volume, direction, n_splits):
        self.start_time = start_time
        self.end_time = end_time
        self.time_seq = split_time_interval(start_time, end_time, n_splits) # notice: 末 - 初, 才是正数
        self.volume = volume
        self.direction = direction
        self.n_splits = n_splits
        self.time_index = 0
        self.remaining_volume = volume  # Track remaining volume to be traded
        
        self.order_placed = False
        self.try_new_vwap = True

        self.total_quantity = volume # 总量
        self.delta_pos = 0 # 当前仓位, 以buy为例。如果为sell，初始quantity = volume
        self.target_delta_pos = 0
        self.cash = 0

def split_time_interval(start_time, end_time, n_splits):
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    interval = (end_seconds - start_seconds) / n_splits

    splits = []
    for i in range(n_splits + 2):
        split_time = seconds_to_time(start_seconds + i * interval)
        splits.append(split_time)
    print(splits)
    return splits
    
def update_period(order):
    start_period = order.time_seq[order.time_index]
    end_period = order.time_seq[order.time_index+1]
    return start_period, end_period

def equal(a, b, error=10):
    return abs(a - b) < error
def search_vwap_opportunity(snapshot, vwap, direction):
    if vwap == 0.0:
        return False
    if direction == 'buy':
        price = snapshot['AskPrice'][0]
        if price <= vwap:
            return True
        else:
            return False
    else:
        price = snapshot['BidPrice'][0]
        if price >= vwap:
            return True
        else:
            return False

def get_private_features(order):
    executed_quantity = order.delta_pos / order.total_quantity
    remaining_time = order.time_index / order.n_splits
    return [executed_quantity, remaining_time]
    
def adjust_price(raw_price):
    '''
    price 要求是正百的int, 如261300
    input: float or int
    output: 符合格式要求的int
    '''
    # 将输入值四舍五入到最近的100
    adjusted_price = round(raw_price / 100) * 100
    return adjusted_price
class BondMomentum(object):

    def __init__(
        self,
        md_api: KellyMdApi,
        
        # orddertail_q: multiprocessing.Queue,
        # tradetail_q: multiprocessing.Queue,
        # del_tra_detail_q: multiprocessing.Queue, 
        snapshot_q: multiprocessing.Queue,
        
        status_selford_q: multiprocessing.Queue,
        selftra_q: multiprocessing.Queue,
        place_selford_q: multiprocessing.Queue
    ):
        self.md_api = md_api
        self.snapshot_q: multiprocessing.Queue = snapshot_q
        self.status_selford_q: multiprocessing.Queue = status_selford_q
        self.selftra_q: multiprocessing.Queue = selftra_q
        self.place_selford_q: multiprocessing = place_selford_q

        self.sig_gen: PlaceOrdSigGen = PlaceOrdSigGen(config.StrategyParam.time_window)
        self.local_ord_id: int = -1
        self.log_folder: str = None
        self.force_quit: bool = False
        self.parent_orders = []
        self.child_orders = []

        self.train_data = []  # 存储 s(market features + private features), a(price, volume), r()
        self.current_state = None 
        self.current_action = None 
        self.market_features = None
        # 初始化log文件
        self.tick_log_file = None
        self.tick_logger = None

        
    def check_data_cache(self):
        len_ = len(self.sig_gen.get_data())
        return len_ == config.StrategyParam.time_window
    
    def get_buffer(self):
        return self.train_data # list[tuple]; tuple: (state: dict, action: dict, reward: float) 
        
    def set_force_quit_flag(
        self,
        force_quit: bool
    ):
        self.force_quit = force_quit
    
    def add_trade_order(self, start_time, end_time, volume, direction, n_splits):
        order = TradeOrder(start_time, end_time, volume, direction, n_splits)
        self.parent_orders.append(order)
        
    def init_work_td(
        self,
        instrument: str
    ):
        if config.StrategyParam.out_log:
            self.log_folder = f'{config.StrategyParam.result_folder}/{config.StrategyParam.name}/{config.StrategyParam.version}/{instrument}/{config.StrategyParam.trading_day}/'
            if not os.path.isdir(self.log_folder):
                os.makedirs(self.log_folder)
            # 初始化tick数据的csv文件
            tick_log_path = os.path.join(self.log_folder, 'tick_data.csv')
            self.tick_o = open(tick_log_path, 'w')
            log = "Time,Volume,Turnover,AskPrice_0,BidPrice_0,VWAP\n"
            self.tick_o.write(log)
            
            file_full_path = f'{self.log_folder}traded_ords.csv'
            self.traded_o = open(file_full_path, 'w')
            log = "localid,direction,volume,price,matchamount,deltapos,matchtime,matchtype\n"
            self.traded_o.write(log)

            file_full_path = f'{self.log_folder}place_ords.csv'
            self.placeord_o = open(file_full_path, 'w')
            log = "time,exchange,ask1,bid1,deltapos,price,volume,direction,ordlocalid\n"
            self.placeord_o.write(log)

        
        if instrument[-1] > 'H':
            self.side_buy = SZ_SIDE_BUY
            self.side_sell = SZ_SIDE_SELL
        else:
            self.side_buy = SH_SIDE_BUY
            self.side_sell = SH_SIDE_SELL
        self.instrument: str = instrument

        self.am_closing_pos: bool = False

        class ThreadSafeCounter(object):
            def __init__(self, init: int = 0):
                self.count: int = init
                self.lock = threading.Lock()

            def increment(self, delta: int):
                with self.lock:
                    self.count += delta

            def get(self) -> int:
                with self.lock:
                    return self.count

            def set(self, init: int = 0):
                with self.lock:
                    self.count = init
        self.q_value_proxy: ThreadSafeCounter = ThreadSafeCounter() # 仓位
        self.snapshot_process = threading.Thread(target=self.__process_snapshotmsg_process, args=(self.md_api, self.snapshot_q))
        self.selftra_process = threading.Thread(target=self.__process_selftramsg_process, args=(self.md_api, self.selftra_q,))
        self.status_selford_process = threading.Thread(target=self.__process_selfordmsg_process, args=(self.md_api, self.status_selford_q,))
        self.snapshot_process.start()
        self.status_selford_process.start()
        self.selftra_process.start()
    def join_work_td(
        self
    ):
        self.snapshot_process.join()
        self.selftra_process.join()
        self.status_selford_process.join()

    def __process_snapshotmsg_process(self, md_api: KellyMdApi, input_q: multiprocessing.Queue):
        if input_q is None or md_api is None:
            return
    
        while not self.force_quit:
            try:
                next_task: dict = input_q.get(True, 0.01)
            except queue.Empty:
                continue
            except Exception:
                return

            current_time = next_task['Time']
            self.market_features = self.sig_gen.add_data(next_task) # features list[feature]
            # print(next_task)
            # print(current_time)
            ask_price_0 = next_task['AskPrice'][0]
            bid_price_0 = next_task['BidPrice'][0]

            for order in self.parent_orders:  # 母单
                order.delta_pos: int = self.q_value_proxy.get() # 持仓
                start_period, end_period = update_period(order)
                # print(start_period, current_time, end_period, order.order_placed)
                if start_period < current_time <= end_period and order.time_index<order.n_splits:
                    if not order.order_placed and self.check_data_cache(): # 如果未挂单
                        ##########按照最优价挂单##################
                        print('按照最优价挂单', current_time)
                        volume_per_split = order.volume // order.n_splits
        
                        if order.direction == 'buy':
                            price = next_task['BidPrice'][0] 
                            direction = KELLY_D_BUY
                        else:
                            price = next_task['AskPrice'][0]
                            direction = KELLY_D_SELL
                        # 记录当前状态
                        private_features = get_private_features(order)
                        self.current_state = {
                            'Time': current_time,
                            'order_local_id': str(self.local_ord_id),
                            'features': self.market_features + private_features
                        }
                        sug_td = TDSug(
                            instrument=next_task['Instrument'], order_localid=str(self.local_ord_id),
                            order_type=KELLY_OT_LIMITPRICE, exchange=next_task['Exchange'],
                            price=price, volume=volume_per_split, direction=direction)
                        if config.StrategyParam.out_log and self.placeord_o.writable():
                            log = f"{next_task['Time']},{sug_td.exchange},{next_task['AskPrice'][0]},{next_task['BidPrice'][0]}," + \
                                  f"{order.delta_pos},{sug_td.price},{sug_td.volume},{sug_td.direction},{sug_td.order_localid}\n"
                            self.placeord_o.write(log)
                            self.placeord_o.flush()
                        # 记录当前动作
                        self.current_action = {
                            'Time': current_time,
                            'order_local_id': str(self.local_ord_id),
                            'volume': volume_per_split,
                            'price': price, 
                            'direction': direction, 
                        }
                        self.local_ord_id -= 1
                        self.__add_selford(sug_td, self.place_selford_q) # add ord 
                        order.remaining_volume -= volume_per_split
                        order.target_delta_pos += volume_per_split

                        if order.remaining_volume < volume_per_split: # TODO: change sth else
                            order.remaining_volume = 0
                        self.__add_eventord(2, self.place_selford_q)
                        order.order_placed = True
                    else:
                        md_api.set_waitevent()
                        ############################################
                    
                    check_deal = equal(order.delta_pos, order.target_delta_pos) # 仓位等于目标仓位

                    if order.order_placed and check_deal:
                        ############### 滑到下一period ##########
                        print('成交,滑到下一period', current_time)
                        order.time_index += 1
                        # order.time_index = min(order.time_index, order.n_splits-1)
                        order.order_placed = False
                    if order.order_placed and (not check_deal) and time_delete(end_period, current_time) < 3*2: # 3s * 3
                        ############# 撤 单 ##############
                        print('one period is comming over, check point is ', order.time_index)
                        sug_td = TDSug(
                            instrument=next_task['Instrument'], 
                            order_localid=str(self.local_ord_id+1),
                                      )
                        self.local_ord_id -= 1
                        self.__add_selford(sug_td, self.place_selford_q) # add cancel ord
                        self.__add_eventord(2, self.place_selford_q)
                        
                        # order.remaining_volume -= volume_per_split
                        if order.remaining_volume < volume_per_split:
                            order.remaining_volume = 0
                        ########## 滑到下一period #################
                        order.order_placed = False
                        order.time_index += 1
                        print('消极挂单, 滑到下一period', current_time)
                else:
                    md_api.set_waitevent()

    def __process_selftramsg_process(
        self,
        md_api: KellyMdApi,
        input_q: multiprocessing.Queue
    ):
        if input_q is None or md_api is None:
            return

        while (not self.force_quit):
            try:
                next_task: dict = input_q.get(True, 0.01)
            except queue.Empty:
                continue
            except Exception:
                return

            if next_task['MatchType'] == KELLY_MT_MATCH: # 这个问题啊啊啊啊啊
                if next_task['Direction'] == KELLY_D_SELL:
                    self.q_value_proxy.increment(-next_task['Volume'])
                elif next_task['Direction'] == KELLY_D_BUY:
                    self.q_value_proxy.increment(next_task['Volume'])
                else:
                    print(f"The trade direction is invalid: {next_task['Direction']}")

            # 计算reward(self.current_state, self.current_action)
            order_local_id = next_task['OrderLocalID']
            vwap, volume = self.market_features[0], next_task['Volume']
            reward = next_task['MatchAmount'] - vwap * volume # sum(n_i * price_i), 成交量 - vwap * volume
            # 将 (state, action, reward) 存入train_data
            if self.current_state is not None and self.current_action is not None:
                self.train_data.append((self.current_state, self.current_action, reward))
                
            if config.StrategyParam.out_log and self.traded_o.writable():
                log = f"{next_task['OrderLocalID']},{'B' if next_task['Direction'] == KELLY_D_BUY else 'S'}," +\
                    f"{next_task['Volume']},{next_task['Price']},{next_task['MatchAmount']},{self.q_value_proxy.get()},{next_task['MatchTime']}," +\
                    f"{'T' if next_task['MatchType'] == KELLY_MT_MATCH else 'D'}\n"
                self.traded_o.write(log)
                self.traded_o.flush()

            md_api.set_waitevent()

    def __process_selfordmsg_process(
        self,
        md_api: KellyMdApi,
        input_q: multiprocessing.Queue
    ):
        if input_q is None:
            return
        while (not self.force_quit):
            try:
                next_task: dict = input_q.get(True, 0.01)
            except queue.Empty:
                continue
            except Exception:
                return
            if len(next_task) > 0:
                md_api.set_waitevent()

    def __add_eventord(
        self,
        wait_event: int,
        input_q: multiprocessing.Queue
    ):
        sug: TDSug = TDSug()
        sug.wait_event = wait_event
        input_q.put(sug)

    def __add_selford(
        self,
        sug_td: TDSug,
        input_q: multiprocessing.Queue
    ):
        input_q.put(sug_td)
