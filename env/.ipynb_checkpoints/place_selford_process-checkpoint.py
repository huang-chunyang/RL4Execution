#!/usr/bin/env python

# encoding: UTF-8

import threading
import queue
import multiprocessing

from kelly_tdapi_imp import KellyTDApiImp


class PlaceSelfOrdProcess(threading.Thread):

    def __init__(
        self,
        session_id: int,
        front_id: int,
        broker_id: str,
        user_id: str,
        td_api: KellyTDApiImp,
        selford_q: multiprocessing.Queue
    ):
        """Constructor"""
        threading.Thread.__init__(self)

        self.input_q: multiprocessing.Queue = selford_q
        self.td_api = td_api
        self.user_id: str = user_id
        self.broker_id: str = broker_id
        self.front_id: int = front_id
        self.session_id: int = session_id

        self.request_id = 10000

        self.cancel_order_local_id: int = 1

        self.force_quit: bool = False

    def set_force_quit_flag(
        self,
        force_quit: bool
    ):
        self.force_quit: bool = force_quit

    def run(self):
        while not self.force_quit:
            try:
                next_task: dict = self.input_q.get(True, 0.01)
                # print(next_task.order_localid, next_task.order_type, next_task.direction, next_task.price, next_task.volume)
            except queue.Empty:
                continue
            except Exception:
                return

            if next_task.wait_event > 0:
                self.td_api.set_waitevent()
            else:
                if next_task.price > 0:
                    order: dict = {}
                    order["Broker"] = self.broker_id
                    order["Account"] = self.user_id
                    order["Exchange"] = next_task.exchange
                    order["Instrument"] = next_task.instrument
                    order["OrderLocalID"] = next_task.order_localid
                    order["OrderType"] = next_task.order_type
                    order["Direction"] = next_task.direction
                    order["Price"] = next_task.price
                    order["Volume"] = next_task.volume
                    self.td_api.req_orderinsert(order, self.request_id)
                    # print('-------------挂单:', next_task.order_localid)
                else:
                    order: dict = {}
                    order["Broker"] = self.broker_id
                    order["Account"] = self.user_id
                    order["FrontID"] = self.front_id
                    order["SessionID"] = self.session_id
                    order["Exchange"] = next_task.exchange
                    order["OrderLocalID"] = next_task.order_localid
                    order["CancelOrderLocalID"] = str(self.cancel_order_local_id)
                    self.cancel_order_local_id += 1
                    self.td_api.req_ordercancel(order, self.request_id)
                    # print('取消挂单:', next_task.order_localid)
                self.request_id += 1
