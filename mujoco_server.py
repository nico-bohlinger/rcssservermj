import sys
import struct
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

        self.mj_ctrl = None
        self.nq = 0
        self.nv = 0


    def handle_client(self, conn, addr):
        print(f"[NEW CONNECTION] {addr} connected.")
        while True:
            team_msg = conn.recv(1).decode("utf-8")
            if team_msg:
                print(f"[{addr}] Received team: {team_msg}")
                conn.send("ACK".encode("utf-8"))
                break
        while True:
            player_id_msg = conn.recv(2).decode("utf-8")
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

        while self.server_running:
            try:
                i = self.connections.index(conn)
                shape_len = struct.unpack('>I', conn.recv(4))[0]
                shape = struct.unpack('>' + 'I'*shape_len, conn.recv(4 * shape_len))
                dtype = conn.recv(10).decode('utf-8').strip()
                data = conn.recv(np.prod(shape) * np.dtype(dtype).itemsize)
                action = np.frombuffer(data, dtype=dtype).reshape(shape)
                self.mj_ctrl[i*self.nu:(i+1)*self.nu] = action
            except Exception as e:
                break
        self.remove_player_list.append((team, player_id))
        conn.close()
        self.connections.remove(conn)
    

    def run_simulation(self):
        xml_path = (Path(__file__).resolve().parent / "data" / "robot.xml").as_posix()
        robot_spec = mujoco.MjSpec()
        robot_spec.from_file(xml_path)
        robot_body = robot_spec.find_body("torso")
        self.nq = 15
        self.nv = 14
        self.nu = 8

        xml_path = (Path(__file__).resolve().parent / "data" / "test.xml").as_posix()
        spec = mujoco.MjSpec()
        spec.from_file(xml_path)
        mj_model = spec.compile()
        mj_data = mujoco.MjData(mj_model)
        nr_substeps = 5
        dt = mj_model.opt.timestep * nr_substeps

        viewer = None if not self.render else MujocoViewer(mj_model, dt)

        while self.server_running:
            if self.add_player_list:
                for team, player_id in self.add_player_list:
                    self.teams[team].append(player_id)
                    print(f"[TEAM {team}] Player {player_id} joined the game.")
                    frame = spec.worldbody.add_frame()
                    frame.attach_body(robot_body, "attached-", f"-{team}_{player_id}")
                    new_robot_body = spec.find_body(f"attached-torso-{team}_{player_id}")
                    new_robot_body.first_geom().rgba = [1, 0, 0, 1] if team == 0 else [0, 0, 1, 1]
                    len_old_qpos = len(mj_data.qpos)
                    len_old_qvel = len(mj_data.qvel)
                    mj_model, mj_data = spec.recompile(mj_model, mj_data)
                    x_sign = 1 if team == 0 else -1
                    init_qpos = np.zeros(self.nq)
                    init_qpos[:3] = self.positions[player_id] * np.array([x_sign, 1, 1])
                    init_qpos[3:7] = [1, 0, 0, 0]
                    init_qvel = np.zeros(self.nv)
                    mj_data.qpos[len_old_qpos:] = init_qpos
                    mj_data.qvel[len_old_qvel:] = init_qvel
                    self.mj_ctrl = np.zeros(mj_model.nu)
                self.add_player_list.clear()
            
            if self.remove_player_list:
                for team, player_id in self.remove_player_list:
                    self.teams[team].remove(player_id)
                    print(f"[TEAM {team}] Player {player_id} left the game.")
                    new_robot_body = spec.find_body(f"attached-torso-{team}_{player_id}")
                    if new_robot_body:
                        spec.detach_body(new_robot_body)
                    mj_model, mj_data = spec.recompile(mj_model, mj_data)
                    self.mj_ctrl = np.zeros(mj_model.nu)
                self.remove_player_list.clear()

            for i, conn in enumerate(self.connections):
                qpos = mj_data.qpos[i*self.nq:(i+1)*self.nq]
                qvel = mj_data.qvel[i*self.nv:(i+1)*self.nv]
                qpos_qvel = np.concatenate([qpos, qvel])
                array_to_send = qpos_qvel.tobytes()
                shape = qpos_qvel.shape
                dtype = qpos_qvel.dtype.name
                conn.sendall(struct.pack('>I', len(shape)) + struct.pack('>' + 'I'*len(shape), *shape))
                conn.sendall(dtype.ljust(10).encode('utf-8'))
                conn.sendall(array_to_send)

            mj_data.ctrl = self.mj_ctrl
            mujoco.mj_step(mj_model, mj_data, nr_substeps)

            if viewer:
                viewer.model = mj_model
                viewer.render(mj_data)
        
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
