import sys
import socket
import threading
import time
import struct
import numpy as np


class Client:
    def __init__(self, host, port, team, player_id):
        self.host = host
        self.port = port
        self.team = team
        self.player_id = player_id
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        self.sock.connect((host, port))
        print("[CONNECTED] Connected to the server.")

    def loop(self):
        self.sock.send(self.team.encode("utf-8"))
        print("[TEAM] Sent the team.")
        while True:
            msg = self.sock.recv(3).decode("utf-8")
            if msg == "ACK":
                break
        self.sock.send(str(self.player_id).zfill(2).encode("utf-8"))
        print("[PLAYER ID] Sent the player ID.")
        while True:
            msg = self.sock.recv(3).decode("utf-8")
            if msg == "ACK":
                break

        while True:
            try:
                shape_len = struct.unpack('>I', self.sock.recv(4))[0]
                shape = struct.unpack('>' + 'I'*shape_len, self.sock.recv(4 * shape_len))
                dtype = self.sock.recv(10).decode('utf-8').strip()
                data = self.sock.recv(np.prod(shape) * np.dtype(dtype).itemsize)
                qpos_qvel = np.frombuffer(data, dtype=dtype).reshape(shape)
                qpos = qpos_qvel[:15]
                qvel = qpos_qvel[15:]

                action = np.random.uniform(-1, 1, 8)
                array_to_send = action.tobytes()
                shape = action.shape
                dtype = action.dtype.name
                self.sock.sendall(struct.pack('>I', len(shape)) + struct.pack('>' + 'I'*len(shape), *shape))
                self.sock.sendall(dtype.ljust(10).encode('utf-8'))
                self.sock.sendall(array_to_send)
            except Exception as e:
                print(f"[ERROR] Connection closed due to: {e}")
                break
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()

    def run(self):
        loop_thread = threading.Thread(target=self.loop)
        loop_thread.start()


if __name__ == "__main__":
    server_host = sys.argv[1]
    server_port = int(sys.argv[2])
    team = sys.argv[3]
    player_id = sys.argv[4]
    client = Client(server_host, server_port, team, player_id)
    client.run()
