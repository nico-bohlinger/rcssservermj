import socket
import threading
from pathlib import Path
import numpy as np
import mujoco

from viewer import MujocoViewer


class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(5)

        self.connections = []
        self.actions = [0.0]

        self.render = True


    def handle_client(self, conn, addr):
        print(f"[NEW CONNECTION] {addr} connected.")
        connected = True
        while connected:
            try:
                msg = conn.recv(32).decode("utf-8")
                if msg:
                    print(f"[{addr}] {msg}")
                    self.actions[0] = float(msg)
                else:
                    connected = False
            except ConnectionResetError:
                connected = False
        print(f"[DISCONNECTION] {addr} disconnected.")
        conn.close()
        self.connections.remove(conn)
    

    def run_simulation(self):
        xml_path = (Path(__file__).resolve().parent / "data" / "ant.xml").as_posix()
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        data.qpos[2] = 0.75
        nr_substeps = 1
        nr_intermediate_steps = 1
        dt = model.opt.timestep * nr_substeps * nr_intermediate_steps

        viewer = None if not self.render else MujocoViewer(model, dt)

        while True:
            for _ in range(nr_intermediate_steps):
                data.ctrl = np.ones(model.nu) * self.actions[0]
                mujoco.mj_step(model, data, nr_substeps)

            if viewer:
                viewer.render(data)


    def run(self):
        try:
            print("[STARTING] Server is starting...")
            thread = threading.Thread(target=self.run_simulation)
            thread.start()

            while True:
                conn, addr = self.sock.accept()
                self.connections.append(conn)

                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.start()
        except:
            print("[SHUTDOWN] Server is shutting down...")
            self.sock.close()


server = Server("localhost", 60000)
server.run()