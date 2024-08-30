import socket
import threading


class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(5)

        self.connections = []
        self.actions = {}

    def handle_client(self, conn, addr):
        print(f"[NEW CONNECTION] {addr} connected.")
        connected = True
        while connected:
            try:
                msg = conn.recv(32).decode("utf-8")
                if msg:
                    print(f"[{addr}] {msg}")
                    self.actions[addr] = msg
                else:
                    connected = False
            except ConnectionResetError:
                connected = False
        print(f"[DISCONNECTION] {addr} disconnected.")
        conn.close()
        self.connections.remove(conn)
    
    def run(self):
        print("[STARTING] Server is starting...")
        while True:
            conn, addr = self.sock.accept()
            self.connections.append(conn)

            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()


server = Server("localhost", 60000)
server.run()