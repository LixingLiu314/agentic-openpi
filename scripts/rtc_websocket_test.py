"""Real WebSocket transport + threaded RTC controller, with in-memory actuators."""
import asyncio
import json
from pathlib import Path
import time

import numpy as np
from openpi_client import msgpack_numpy
from websockets.asyncio.server import serve
from websockets.sync.client import connect

from piper_rtc import execute_rtc, PROTOCOL
from serve_recurrent_subtask_piper import RecurrentWebsocketServer


class TestPolicy:
    def __init__(self):self.query=0;self.origin=0
    def new_session(self):return TestPolicy()
    def reset(self):self.query=0;self.origin=0
    def infer(self,obs):
        req=obs["rtc"];assert req["query_id"]==self.query+1
        if self.query:
            assert req["previous_query_id"]==self.query
            self.origin+=req["consumed_steps"]
            time.sleep(.1)
        self.query+=1
        actions=np.repeat((self.origin+np.arange(50))[:,None],14,axis=1).astype(np.float32)/1000
        return dict(actions=actions,rtc=dict(req),subtask="reach",session_query=self.query)


async def main():
    root=TestPolicy();service=RecurrentWebsocketServer(root,prompt="task",metadata={"rtc_protocol":PROTOCOL})
    async with serve(service._handler,"127.0.0.1",0,compression=None) as server:
        url="ws://127.0.0.1:%d"%server.sockets[0].getsockname()[1]
        def client():
            pack=msgpack_numpy.Packer();commands=[];times=[];records=[]
            with connect(url,compression=None) as ws:
                assert msgpack_numpy.unpackb(ws.recv())["rtc_protocol"]==PROTOCOL
                def request(job):
                    ws.send(pack.pack(dict(job["observation"],rtc=job["rtc"])))
                    return msgpack_numpy.unpackb(ws.recv(timeout=5))
                def publish(action):commands.append(action.copy());times.append(time.monotonic())
                report=execute_rtc(request=request,snapshot=lambda:({"state":np.zeros(14)},{}),publish=publish,
                    record=lambda job,result,skipped,total:records.append((result["session_query"],skipped)),
                    action_sent=lambda *x:None,check=lambda:None,stopped=lambda:False,max_steps=65)
            np.testing.assert_allclose(np.asarray(commands)[:,0],np.arange(65)/1000,atol=1e-7)
            assert np.diff(times).max()<.095
            assert records[1][1]>=3
            with connect(url,compression=None) as fresh:
                fresh.recv();fresh.send(pack.pack({"rtc":dict(protocol=PROTOCOL,query_id=1,
                    previous_query_id=None,consumed_steps=0,delay_steps=0)}))
                assert msgpack_numpy.unpackb(fresh.recv())["session_query"]==1
            return dict(passed=True,real_websocket=True,hardware_calls=False,steps=len(commands),
                        queries=len(records),max_gap_ms=float(np.diff(times).max()*1000),fresh_session=True)
        result=await asyncio.to_thread(client)
    assert root.query==0
    print(json.dumps(result))


if __name__=="__main__":asyncio.run(main())
