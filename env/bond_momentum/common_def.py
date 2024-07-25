class TDSug(object):
    def __init__(
        self,
        wait_event: int = 0,
        instrument: str = "",
        order_localid: str = "0",
        order_type: int = 0,
        exchange: int = 0,
        price: int = 0,
        volume: int = 0,
        direction: int = 0

    ):
        self.wait_event = wait_event
        self.instrument = instrument
        self.order_localid = order_localid
        self.order_type = order_type
        self.exchange = exchange
        self.price = price
        self.volume = volume
        self.direction = direction