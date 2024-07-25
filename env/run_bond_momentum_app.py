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


def ConnectMDClient(
    snapshot_q: multiprocessing.Queue,
    time_stamp: int,
    data_finish_event: BoundedSemaphore
) -> KellyMDApiImp:
    front_address = config.MDConfig.front_address
    user = config.MDConfig.user
    password = config.MDConfig.password
    trading_day = config.StrategyParam.trading_day
    inst = config.StrategyParam.instrument

    md_api = KellyMDApiImp(AM_BEGIN, AM_END, PM_BEGIN, PM_END1, None, None, None,
                           snapshot_q, None, None, data_finish_event)
    md_api.create_kelly_mdapi("")

    md_api.register_front(front_address)

    request_id = 1
    md_api.req_userlogin(user, password, trading_day, request_id, time_stamp)
    request_id += 1
    ticker_list = [{'Instrument': inst}]
    print(ticker_list)
    md_api.sub_snapshot(ticker_list, len(ticker_list), request_id)

    # e_none = 0,
    # e_snapshot = 1,
    # e_orgord = 2,
    # e_orgtra = 4,
    # e_rebuildob = 8,
    # e_rebuildoq = 16,
    # e_selftra = 32,
    # e_selford = 64,
    # e_time_interval = 128,
    wait_mode = KELLY_WAIT_SNAPSHOT
    md_api.set_waitmode(wait_mode)

    # breakpoint()
    ret = md_api.init()
    if ret == REQUESTDATA_INIT or ret == REQUESTDATA_PREPARED:
        data_finish_event.acquire()
    else:
        md_api = None

    return md_api

def ConnectTDClient(
    selford_q: multiprocessing.Queue,
    selftra_q: multiprocessing.Queue,
    time_stamp: int,
    data_finish_event: BoundedSemaphore
) -> KellyTDApiImp:
    front_address = config.TDConfig.front_address
    user = config.TDConfig.user
    password = config.TDConfig.password
    trading_day = config.StrategyParam.trading_day

    td_api = KellyTDApiImp(selford_q, selftra_q, data_finish_event)
    td_api.create_kelly_tdapi("")

    request_id = 1
    td_api.register_front(front_address)
    td_api.req_userlogin(user, password, trading_day, request_id, time_stamp)
    request_id += 1
    
    td_api.subscribe_publictopic(0) 

    # e_none = 0,
    # e_snapshot = 1,
    # e_orgord = 2,
    # e_orgtra = 4,
    # e_rebuildob = 8,
    # e_rebuildoq = 16,
    # e_selftra = 32,
    # e_selford = 64,
    # e_time_interval = 128,
    wait_mode = KELLY_WAIT_ORDERFIELD | KELLY_WAIT_TRADEFIELD
    td_api.set_waitmode(wait_mode)

    if td_api.init() == 0:
        data_finish_event.acquire()
    else:
        td_api = None

    return td_api

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
    # print('--------------------------')
    # print(time_stamp)
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

    snapshot_q = multiprocessing.Queue()
    status_selford_q = multiprocessing.Queue()
    selftra_q = multiprocessing.Queue()
    place_selford_q = multiprocessing.Queue()
    data_finish_event = BoundedSemaphore(2)

    if config.StrategyParam.out_log:
        param = config.StrategyParam
        log_folder = f'{param.result_folder}/{param.name}/{param.version}/{param.instrument}/{param.trading_day}/'
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)
        file_full_path = f'{log_folder}strategy_param.csv'
        strategy_param_o = open(file_full_path, 'w')
        log = "max_window,ordbaseunit_volume,ordmaxunit_volume\n"
        strategy_param_o.write(log)
        log = f"{param.max_window},{param.ordbaseunit_volume},{param.ordmaxunit_volume}\n"
        strategy_param_o.write(log)
        strategy_param_o.flush()

    # breakpoint()
    td_api: KellyTDApiImp = ConnectTDClient(status_selford_q, selftra_q, app_id, data_finish_event)
    print('---------------------def TD_api----------------------')
    spread_trading = BondMomentum(td_api, snapshot_q, status_selford_q, selftra_q, place_selford_q)

    split_num = min(args.volume//100, (args.end_time-args.start_time)//3)
    print('split_num:', split_num)
    spread_trading.add_trade_order(args.start_time, args.end_time, args.volume, args.direction, n_splits=split_num)
    print('---------------------def BondMomentum----------------------')
    spread_trading.init_work_td(config.StrategyParam.instrument)
    print('-----------------------spread_trading.init_work_td------------')
    place_selford_process = PlaceSelfOrdProcess(0, 0, config.TDConfig.broker_id, config.TDConfig.user, td_api, place_selford_q)
    print('---------------------self ord process -----------------------')
    place_selford_process.start()

    md_api = ConnectMDClient(snapshot_q, app_id, data_finish_event) # read snapshot (market data)

    print("The application start successfully")

    data_finish_event.acquire()

    md_api.join()
    md_api.release()
    td_api.join()
    td_api.release()
    spread_trading.set_force_quit_flag(True)
    place_selford_process.set_force_quit_flag(True)

    spread_trading.join_work_td()
    place_selford_process.join()

    sys.exit(0)
