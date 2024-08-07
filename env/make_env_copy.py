#!/usr/bin/env python

import sys
import os
import argparse
import config
import multiprocessing
from datetime import datetime
from threading import BoundedSemaphore

from kelly_mdapi_imp import KellyMDApiImp
from kelly_tdapi_imp import KellyTDApiImp
from simkelly_pyapi.kelly_pymd_api import (KELLY_WAIT_SNAPSHOT,
                                           REQUESTDATA_DATE_NOAUTHORITY,
                                           REQUESTDATA_TICKER_NOAUTHORITY,
                                           REQUESTDATA_LOADFAILED,
                                           REQUESTDATA_INVALID_USER,
                                           REQUESTDATA_INVALID_PASSWORD,
                                           REQUESTDATA_INIT,
                                           REQUESTDATA_PREPARED
                                           )
from simkelly_pyapi.kelly_pytd_api import (KELLY_WAIT_ORDERFIELD,
                                           KELLY_WAIT_TRADEFIELD
                                           )

from bond_momentum.bond_momentum import (BondMomentum, AM_BEGIN, AM_END, PM_BEGIN, PM_END1)
from place_selford_process import PlaceSelfOrdProcess
from utils import time_delete, time_to_seconds

class TradingEnv:
    def __init__(self, config, time_stamp): # TODO: add policy here

        self.time_stamp = time_stamp
        
        self.config = config
        
        self.snapshot_q = multiprocessing.Queue()
        self.status_selford_q = multiprocessing.Queue()
        self.selftra_q = multiprocessing.Queue()
        self.place_selford_q = multiprocessing.Queue()
        self.data_finish_event = BoundedSemaphore(2)
        
        self.td_api = self.connect_td_client()
        # self.md_api = self.connect_md_client()

        self.spread_trading = BondMomentum(self.td_api, self.snapshot_q, self.status_selford_q, self.selftra_q, self.place_selford_q)
        
        self.spread_trading.add_trade_order(config.TradeParam.start_time, config.TradeParam.end_time, config.TradeParam.volume, config.TradeParam.direction, n_splits=config.TradeParam.split_num)
        self.spread_trading.init_work_td(config.StrategyParam.instrument)
        
        self.place_selford_process = PlaceSelfOrdProcess(0, 0, config.TDConfig.broker_id, config.TDConfig.user, self.td_api, self.place_selford_q)
        self.place_selford_process.start()

    def connect_md_client(self) -> KellyMDApiImp:
        
        front_address = self.config.MDConfig.front_address
        user = self.config.MDConfig.user
        password = self.config.MDConfig.password
        trading_day = self.config.StrategyParam.trading_day
        inst = self.config.StrategyParam.instrument

        md_api = KellyMDApiImp(AM_BEGIN, AM_END, PM_BEGIN, PM_END1, None, None, None,
                               self.snapshot_q, None, None, self.data_finish_event)
        md_api.create_kelly_mdapi("")

        md_api.register_front(front_address)

        request_id = 1
        md_api.req_userlogin(user, password, trading_day, request_id, int(self.time_stamp))
        request_id += 1
        ticker_list = [{'Instrument': inst}]
        print(ticker_list)
        md_api.sub_snapshot(ticker_list, len(ticker_list), request_id)

        wait_mode = KELLY_WAIT_SNAPSHOT
        md_api.set_waitmode(wait_mode)

        ret = md_api.init()
        if ret == REQUESTDATA_INIT or ret == REQUESTDATA_PREPARED:
            self.data_finish_event.acquire()
        else:
            md_api = None

        return md_api

    def connect_td_client(self) -> KellyTDApiImp:
        front_address = self.config.TDConfig.front_address
        user = self.config.TDConfig.user
        password = self.config.TDConfig.password
        trading_day = self.config.StrategyParam.trading_day

        td_api = KellyTDApiImp(self.status_selford_q, self.selftra_q, self.data_finish_event)
        td_api.create_kelly_tdapi("")

        request_id = 1
        td_api.register_front(front_address)
        print(user, password, trading_day, request_id, self.time_stamp)
        td_api.req_userlogin(user, password, trading_day, request_id, int(self.time_stamp))
        request_id += 1

        td_api.subscribe_publictopic(0)

        wait_mode = KELLY_WAIT_ORDERFIELD | KELLY_WAIT_TRADEFIELD
        td_api.set_waitmode(wait_mode)

        if td_api.init() == 0:
            self.data_finish_event.acquire()
        else:
            td_api = None

        return td_api

    def reset(self, policy, parent_order, info):
        # 模拟交易逻辑，根据policy执行交易行为
        state = self.get_state()
        action = policy(state)
        reward, next_state = self.execute_trade(action, parent_order, info)
        return state, action, reward, next_state

    def trade(self):
        # 按照给定的policy，对母单进行交易，一整天
        # return s, a, r, s'
        self.md_api = self.connect_md_client() # 开始交易的！！！！！
        
        print("The application start successfully")
        
        self.data_finish_event.acquire()
        print('get finish event')
        self.md_api.join()
        self.md_api.release()
        self.td_api.join()
        self.td_api.release()
        self.spread_trading.set_force_quit_flag(True)
        self.place_selford_process.set_force_quit_flag(True)
    
        self.spread_trading.join_work_td()
        self.place_selford_process.join()
        return self.spread_trading.get_buffer()
    def exit(self):
        sys.exit(0)

    def execute_trade(self, action, parent_order, info):
        # 执行交易并计算奖励
        # 这里根据具体需求实现交易执行和奖励计算逻辑
        reward = 0
        next_state = self.get_state()
        return reward, next_state

if __name__ == '__main__':
    parser = argparse.ArgumentParser("parameter")
    parser.add_argument('-version', '--version', default=None, type=str, help="specify the version of the stragety")
    parser.add_argument('-inst', '--inst', default=None, type=str, help="the instrument code of testing data")
    parser.add_argument('-td', '--td', default=None, type=str, help="the trading date code of testing data: '20221111' keep len equal to 8")
    parser.add_argument('-logdir', '--logdir', default=None, type=str, help="the output log directory")
    parser.add_argument('-mdtdfront', '--mdtdfront', default=None, type=str, help="The ipaddress and port of md front")
    parser.add_argument('-tdfront', '--tdfront', default=None, type=str, help="The ipaddress and port of td front")

    parser.add_argument('-start_time', '--start_time', default=93000000, type=int, help="The start time of OE")
    parser.add_argument('-end_time', '--end_time', default=113000000, type=int, help="The end time of OE")
    parser.add_argument('-volume', '--volume', default=1000, type=int, help="The volume of OE")
    parser.add_argument('-direction', '--direction', default='buy', type=str, help="The direction of OE")
    args = parser.parse_args()

    current_time = datetime.now()
    time_stamp = current_time.timestamp()
    app_id = int(time_stamp * 10000000)

    if args.version is not None:
        config.StrategyParam.version = args.version
    if args.inst is not None:
        config.StrategyParam.instrument = args.inst
        print(args.inst)
    if args.td is not None:
        config.StrategyParam.trading_day = args.td
    if args.logdir is not None:
        config.StrategyParam.result_folder = args.logdir
    if args.mdtdfront is not None:
        config.MDConfig.front_address = args.mdtdfront
    if args.tdfront is not None:
        config.TDConfig.front_address = args.tdfront
        
    param = config.StrategyParam
    log = f"The application starting with parameter [inst: {param.instrument}, td: {param.trading_day}, " +\
        f"log_dir: {param.result_folder}, appid: {app_id}"
    print(log)
    # trade params 
    split_num = int(min(config.TradeParam.volume//100, time_delete(config.TradeParam.end_time, config.TradeParam.start_time)//(3*20)))
    config.TradeParam.split_num = split_num
    print('split_num:', split_num)
    env = TradingEnv(config, time_stamp)
    train_data = env.trade()
    print('train_data', len(train_data))
    # print(train_data)
    for sample in train_data:
        print(sample, '\n')
    env.exit()
    # # 示例：调用模拟交易函数
    # policy = lambda state: None  # 假设这是一个简单的策略函数
    # parent_order = None
    # info = None
    # state, action, reward, next_state = env.sim_trade(policy, parent_order, info)
    # print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

    # sys.exit(0)
