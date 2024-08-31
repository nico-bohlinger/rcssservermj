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

        self.teams = {
            0: [],
            1: [],
        }
        self.player_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        z_pos = 1.2
        self.positions = {
            1: [29.0, 0.0, z_pos],
            2: [22.0, 12.0, z_pos],
            3: [22.0, 4.0, z_pos],
            4: [22.0, -4.0, z_pos],
            5: [22.0, -12.0, z_pos],
            6: [15.0, 0.0, z_pos],
            7: [4.0, 16.0, z_pos],
            8: [11.0, 6.0, z_pos],
            9: [11.0, -6.0, z_pos],
            10: [4.0, -16.0, z_pos],
            11: [7.0, 0.0, z_pos],
        }

        self.add_player_list = []
        self.remove_player_list = []


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
        
        self.add_player_list.append((team, player_id))

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
        self.remove_player_list.append((team, player_id))
        conn.close()
        self.connections.remove(conn)
    

    def run_simulation(self):
        xml_path = (Path(__file__).resolve().parent / "data" / "test.xml").as_posix()
        spec = mujoco.MjSpec()
        spec.from_file(xml_path)
        model = spec.compile()
        data = mujoco.MjData(model)
        nr_substeps = 1
        nr_intermediate_steps = 1
        dt = model.opt.timestep * nr_substeps * nr_intermediate_steps

        viewer = None if not self.render else MujocoViewer(model, dt)

        while self.server_running:
            if self.add_player_list:
                for team, player_id in self.add_player_list:
                    self.teams[team].append(player_id)
                    print(f"[TEAM {team}] Player {player_id} joined the game.")
                    body_name = f"team_{team}__player_{player_id}"
                    body = spec.find_body(body_name)
                    if not body:
                        body = spec.worldbody.add_body()
                    body.name = body_name
                    geom = body.add_geom()
                    x_sign = 1 if team == 0 else -1
                    geom.size = [0.2, 0.4, 1.0]
                    geom.rgba = np.array([1.0, 0.0, 0.0, 1.0]) if team == 0 else np.array([0.0, 0.0, 1.0, 1.0])
                    geom.type = mujoco.mjtGeom.mjGEOM_BOX
                    joint = body.add_joint()
                    joint.type = mujoco.mjtJoint.mjJNT_FREE
                    len_old_qpos = len(data.qpos)
                    model, data = spec.recompile(model, data)
                    data.qpos[0 + len_old_qpos: 3 + len_old_qpos] = self.positions[player_id] * np.array([x_sign, 1, 1])
                self.add_player_list.clear()
            
            if self.remove_player_list:
                for team, player_id in self.remove_player_list:
                    self.teams[team].remove(player_id)
                    print(f"[TEAM {team}] Player {player_id} left the game.")
                    body_name = f"team_{team}__player_{player_id}"
                    body = spec.find_body(body_name)
                    if body:
                        body.first_geom().delete()
                        body.first_joint().delete()
                    model, data = spec.recompile(model, data)
                self.remove_player_list.clear()

            for _ in range(nr_intermediate_steps):
                mujoco.mj_step(model, data, nr_substeps)

            if viewer:
                viewer.model = model
                viewer.render(data)
        
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
