import sys
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
        self.server_running = False

        self.render = True

        self.spec = None
        self.model = None
        self.data = None

        self.teams = {
            0: [],
            1: [],
        }
        self.player_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


    def handle_client(self, conn, addr):
        print(f"[NEW CONNECTION] {addr} connected.")
        while True:
            team_msg = conn.recv(32).decode("utf-8")
            if team_msg:
                print(f"[{addr}] Received team: {team_msg}")
                conn.send("ACK".encode("utf-8"))
                break
        while True:
            player_id_msg = conn.recv(32).decode("utf-8")
            if player_id_msg:
                print(f"[{addr}] Received player ID: {player_id_msg}")
                conn.send("ACK".encode("utf-8"))
                break

        team = int(team_msg)
        player_id = int(player_id_msg)
        is_viable_team = team in self.teams
        is_viable_player_id = player_id in self.player_ids
        is_player_id_not_taken = player_id not in self.teams[team]
        if not is_viable_team or not is_viable_player_id or not is_player_id_not_taken:
            conn.send("Invalid team or player ID.".encode("utf-8"))
            conn.close()
            return
        
        self.teams[team].append(player_id)
        print(f"[TEAM {team}] Player {player_id} joined the game.")

        body_name = f"team_{team}__player_{player_id}"
        body = self.spec.find_body(body_name)
        if not body:
            body = self.spec.worldbody.add_body()
        body.name = body_name
        geom = body.add_geom()
        x_sign = 1 if team == 0 else -1
        geom.pos = [x_sign * 3.0 * player_id, 0.0, 1.2]
        geom.size = [0.2, 0.4, 1.0]
        geom.rgba = np.array([1.0, 0.0, 0.0, 1.0]) if team == 0 else np.array([0.0, 0.0, 1.0, 1.0])
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        # joint = body.add_joint()
        # joint.type = mujoco.mjtJoint.mjJNT_FREE
        self.model, self.data = self.spec.recompile(self.model, self.data)

        connected = True
        while connected and self.server_running:
            try:
                msg = conn.recv(32).decode("utf-8")
                if msg:
                    print(f"[{addr}] {msg}")
                else:
                    connected = False
            except ConnectionResetError:
                connected = False
        print(f"[TEAM {team}] Player {player_id} left the game.")
        self.teams[team].remove(player_id)
        geom.delete()
        # joint.delete()
        self.model, self.data = self.spec.recompile(self.model, self.data)
        conn.close()
        self.connections.remove(conn)
    

    def run_simulation(self):
        xml_path = (Path(__file__).resolve().parent / "data" / "test.xml").as_posix()
        self.spec = mujoco.MjSpec()
        self.spec.from_file(xml_path)

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        nr_substeps = 1
        nr_intermediate_steps = 1
        dt = self.model.opt.timestep * nr_substeps * nr_intermediate_steps

        viewer = None if not self.render else MujocoViewer(self.model, dt)

        while self.server_running:
            for _ in range(nr_intermediate_steps):
                mujoco.mj_step(self.model, self.data, nr_substeps)

            if viewer:
                viewer.model = self.model
                viewer.render(self.data)
        
        if viewer:
            viewer.close()
            viewer.stop()


    def run(self):
        try:
            print("[STARTING] Server is starting...")
            self.server_running = True
            sim_thread = threading.Thread(target=self.run_simulation)
            sim_thread.start()

            while self.server_running:
                conn, addr = self.sock.accept()
                self.connections.append(conn)

                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.start()
        except:
            print("[SHUTDOWN] Server is shutting down...")
            self.server_running = False
            for conn in self.connections:
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()


if __name__ == "__main__":
    host = sys.argv[1]
    port = int(sys.argv[2])
    server = Server(host, port)
    server.run()
