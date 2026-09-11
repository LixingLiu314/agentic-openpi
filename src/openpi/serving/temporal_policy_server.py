"""One R2 history per websocket; shared weights, independent session/reset state."""
import logging
import time
import traceback

from openpi_client import msgpack_numpy
import websockets
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class TemporalPolicyServer(WebsocketPolicyServer):
    async def _handler(self,websocket):
        policy=self._policy.new_session()
        packer=msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        try:
            async for payload in websocket:
                try:
                    observation=msgpack_numpy.unpackb(payload)
                    begun=time.monotonic()
                    action=policy.infer(observation)
                    action['server_timing']={'infer_ms':(time.monotonic()-begun)*1000}
                    await websocket.send(packer.pack(action))
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(code=1011,reason='Invalid request or policy failure; start a fresh session')
                    break
        except websockets.ConnectionClosed:
            logging.info('Temporal session connection closed')
        finally:
            policy.reset()
