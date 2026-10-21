"""Exercise the HTTP handler with in-memory request and response bytes."""

from io import BytesIO
import json
import socket
from types import SimpleNamespace

from tensorfold.server.http import make_handler


LOOPBACK = SimpleNamespace(server_address=("127.0.0.1", 0))   # the server a handler answers for: a loopback bind


def post(app, body, path="/v1/chat/completions"):
    payload = json.dumps(body).encode()
    incoming = (f"POST {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload

    class Connection:
        def __init__(self):
            self.output = bytearray()

        def makefile(self, *args):
            return BytesIO(incoming)

        def sendall(self, data):
            self.output.extend(data)

        def fileno(self):
            return connected.fileno()

        def recv(self, *args):
            return connected.recv(*args)

    connection = Connection()
    connected, peer = socket.socketpair()
    try:
        make_handler(app)(connection, ("127.0.0.1", 0), LOOPBACK)
    finally:
        connected.close()
        peer.close()
    headers, response = bytes(connection.output).split(b"\r\n\r\n", 1)
    return int(headers.split()[1]), response.decode()
