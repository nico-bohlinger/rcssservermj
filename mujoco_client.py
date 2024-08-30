import sys
import socket
import threading
import time


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
            msg = self.sock.recv(32).decode("utf-8")
            if msg == "ACK":
                break
        self.sock.send(self.player_id.encode("utf-8"))
        print("[PLAYER ID] Sent the player ID.")
        while True:
            msg = self.sock.recv(32).decode("utf-8")
            if msg == "ACK":
                break

        while True:
            try:
                time.sleep(1)
                self.sock.send("Hello, server!".encode("utf-8"))
            except:
                print("[ERROR] Connection closed.")
                break

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
