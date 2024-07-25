#!/usr/bin/env python

# encoding: UTF-8

import multiprocessing
from threading import BoundedSemaphore

from simkelly_pyapi.kelly_pymd_api import KellyMdApi


class KellyMDApiImp(KellyMdApi):

    def __init__(
        self,
        am_starttime: int,
        am_endtime: int,
        pm_starttime: int,
        pm_endtime: int,
        orddertail_q: multiprocessing.Queue,
        tradetail_q: multiprocessing.Queue,
        del_tra_detail_q: multiprocessing.Queue,
        snapshot_q: multiprocessing.Queue,
        rebuildob_q: multiprocessing.Queue,
        ordqueue_q: multiprocessing.Queue,
        data_finish_event: BoundedSemaphore
    ):
        """Constructor"""
        super(KellyMDApiImp, self).__init__()

        self.orddertail_q: multiprocessing.Queue = orddertail_q
        self.tradetail_q: multiprocessing.Queue = tradetail_q
        self.del_tra_detail_q: multiprocessing.Queue = del_tra_detail_q
        self.snapshot_q: multiprocessing.Queue = snapshot_q
        self.rebuildob_q: multiprocessing.Queue = rebuildob_q
        self.ordqueue_q: multiprocessing.Queue = ordqueue_q

        self.am_starttime: int = am_starttime
        self.am_endtime: int = am_endtime
        self.pm_starttime: int = pm_starttime
        self.pm_endtime: int = pm_endtime

        self.data_finish_event = data_finish_event

    def on_frontdisconnected(self):
        """"""
        pass

    def on_rspuserlogin(self, data, error, request_id, last):
        """"""
        pass

    def on_rspqryinstrument(self, data, error, request_id, last):
        """"""
        pass

    def on_rspsubsnapshot(self, data, error, request_id, last):
        """"""
        pass

    def on_rspunsubsnapshot(self, data, error, request_id, last):
        """"""
        pass

    def on_rspsubdetail(self, data, error, request_id, last):
        """"""
        pass

    def on_rspunsubdetail(self, data, error, request_id, last):
        """"""
        pass

    def on_rtnsnapshot(self, data):
        """"""
        if (data["Time"] < self.am_starttime or
            data["Time"] > self.pm_endtime or
            (data["Time"] > self.am_endtime and data["Time"] < self.pm_starttime)
        ):
            super().set_waitevent()
        else:
            self.snapshot_q.put(data)
            'receive snapshot data'

    def on_rtnorderdetail(self, data):
        """"""
        self.orddertail_q.put(data)

    def on_rtntradedetail(self, data):
        """"""
        if (data["Time"] < self.am_starttime or
            data["Time"] > self.pm_endtime or
            (data["Time"] > self.am_endtime and data["Time"] < self.pm_starttime)
        ):
            super().set_waitevent()
        else:
            if data["Price"] == 0:
                self.del_tra_detail_q.put(data)
            self.tradetail_q.put(data)

    def on_rtnrebuildob(self, data):
        """"""
        if (data["Time"] < self.am_starttime or
            data["Time"] > self.pm_endtime or
            (data["Time"] > self.am_endtime and data["Time"] < self.pm_starttime)
        ):
            super().set_waitevent()
        else:
            self.rebuildob_q.put(data)

    def on_rtnorderqueue(self, data):
        """"""
        if (data["Time"] < self.am_starttime or
            data["Time"] > self.pm_endtime or
            (data["Time"] > self.am_endtime and data["Time"] < self.pm_starttime)
        ):
            super().set_waitevent()
        else:
            self.ordqueue_q.put(data)

    def on_subtimeinterval(self):
        """"""
        pass

    def on_subdatafinish(self):
        """"""
        print("KellyMDApiImp receive the data finish event")
        self.data_finish_event.release()
