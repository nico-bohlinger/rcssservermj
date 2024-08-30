import socket
import threading


class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        print("[CONNECTED] Connected to the server.")

    def receive_messages(self):
        while True:
            try:
                msg = self.sock.recv(32).decode("utf-8")
                if msg:
                    print(msg)
            except ConnectionResetError:
                print("[ERROR] Connection closed by the server.")
                break
            except:
                print("[ERROR] An error occurred.")
                break

    def send_messages(self):
        while True:
            msg = input("Type a message: ")
            try:
                self.sock.send(msg.encode("utf-8"))
            except:
                print("[ERROR] An error occurred while sending the message.")
                break

    def run(self):
        receive_thread = threading.Thread(target=self.receive_messages)
        receive_thread.start()

        send_thread = threading.Thread(target=self.send_messages)
        send_thread.start()


server_host = "localhost"
server_port = 60000
client = Client(server_host, server_port)
client.run()
