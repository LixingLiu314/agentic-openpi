import asyncio
import numpy as np
import websockets
from openpi_client import msgpack_numpy
from openpi.serving.temporal_policy_server import TemporalPolicyServer


class Policy:
    def __init__(self):self.children=[];self.count=0;self.closed=False
    def new_session(self):
        child=Policy();self.children.append(child);return child
    def infer(self,obs):
        if obs.get('bad'):raise ValueError('bad session')
        self.count+=1;return {'count':self.count,'actions':np.full((50,14),self.count,dtype=np.float32)}
    def reset(self):self.closed=True;self.count=0


class Socket:
    def __init__(self,rows):self.rows=rows;self.sent=[];self.closed=None
    async def send(self,value):self.sent.append(value)
    async def close(self,**kwargs):self.closed=kwargs
    def __aiter__(self):return self
    async def __anext__(self):
        if not self.rows:raise StopAsyncIteration
        return msgpack_numpy.Packer().pack(self.rows.pop(0))


def test_connections_are_independent_and_released():
    root=Policy();server=TemporalPolicyServer(root,metadata={'variant':'r2'})
    a=Socket([{},{}]);b=Socket([{}])
    asyncio.run(server._handler(a));asyncio.run(server._handler(b))
    assert [msgpack_numpy.unpackb(x)['count'] for x in a.sent[1:]]==[1,2]
    assert msgpack_numpy.unpackb(b.sent[1])['count']==1
    assert all(c.closed for c in root.children) and root.count==0


def test_errors_close_and_clear_state():
    root=Policy();server=TemporalPolicyServer(root);socket=Socket([{}, {'bad':True}, {}])
    asyncio.run(server._handler(socket))
    assert socket.closed['code']==1011 and root.children[0].closed
    assert 'bad session' in socket.sent[-1]


def test_real_loopback_transport_preserves_arrays_and_new_connections():
    async def run():
        root=Policy();server=TemporalPolicyServer(root,metadata={'variant':'r2'})
        pack=msgpack_numpy.Packer()
        async with websockets.serve(server._handler,'127.0.0.1',0) as listener:
            port=listener.sockets[0].getsockname()[1]
            for requests in [2,1]:
                async with websockets.connect(f'ws://127.0.0.1:{port}') as client:
                    assert msgpack_numpy.unpackb(await client.recv())['variant']=='r2'
                    for count in range(1,requests+1):
                        await client.send(pack.pack({'session':{'mode':'offline'}}))
                        output=msgpack_numpy.unpackb(await client.recv())
                        assert output['count']==count
                        assert output['actions'].shape==(50,14) and output['actions'].dtype==np.float32
                        assert np.all(output['actions']==count)
        assert len(root.children)==2 and all(c.closed for c in root.children)
    asyncio.run(run())
