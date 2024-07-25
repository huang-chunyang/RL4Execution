#!/usr/bin/env python

# encoding: UTF-8

import multiprocessing
from threading import BoundedSemaphore

from simkelly_pyapi.kelly_pytd_api import KellyTdApi


class KellyTDApiImp(KellyTdApi):

    def __init__(
        self,
        selford_q: multiprocessing.Queue,
        selftra_q: multiprocessing.Queue,
        data_finish_event: BoundedSemaphore
    ):
        """Constructor"""
        super(KellyTDApiImp, self).__init__()

        self.selford_q = selford_q
        self.selftra_q = selftra_q

        self.data_finish_event = data_finish_event

    def on_frontconnected(self):
        """"""
        pass

    def on_frontdisconnected(self):
        """"""
        pass

    def on_heartbeatwarning(self, timelapse):
        """"""
        pass

    def on_rspuserlogin(self, data, error, request_id, last):
        """"""
        pass

    def on_rspuserlogout(self, error, request_id, last):
        """"""
        pass

    def on_rsperror(self, error, request_id, last):
        """"""
        pass

    def on_rtnorder(self, data):
        """"""
        self.selford_q.put(data)

    def on_rtntrade(self, data):
        """"""
        self.selftra_q.put(data)

    def on_rsporderinsert(self, data, error, request_id, last):
        """"""
        pass

    def on_rspbatchorderinsert(self, base_data, orders, error, request_id, last):
        """"""
        pass

    def on_rspordercancel(self, data, error, request_id, last):
        """"""
        pass

    def on_rspbatchordercancel(self, base_data, orders, error, request_id, last):
        """"""
        pass

    def on_errrtnorderinsert(self, orders, error):
        """"""
        pass

    def on_errrtnordercancel(self, orders, error):
        """"""
        pass

    def on_subdatafinish(self):
        """"""
        print("KellyTDApiImp receive the data finish event")
        self.data_finish_event.release()
