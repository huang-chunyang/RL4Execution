class MDConfig(object):
    front_address: str = 'hptm.aitopia.tech'
    user: str = 'huangchunyang'
    password: str = 'huangchunyang'


class TDConfig(object):
    broker_id: str = ''
    front_address: str = 'hptm.aitopia.tech'
    user: str = 'huangchunyang'
    password: str = 'huangchunyang'


class StrategyParam(object):
    out_log: bool = True
    version: str = 'tag_1'
    name: str = 'strategy'
    result_folder: str = './logs/'
    trading_day: str = '20230829'
    instrument: str = '600566.SH'
    time_window: int = 20
    ordbaseunit_volume: int = 100
    ordmaxunit_volume: int = ordbaseunit_volume * 7

class TradeParam(object):
    start_time: int =  93100000
    end_time:   int = 100000000
    volume: int = int(1e4)
    direction: str = 'buy'
    split_num: int = 15
    policy_path: str = ''
