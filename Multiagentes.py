import agentpy as ap
import socket
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Agente Auto
# -----------------------------
class VehicleAgent(ap.Agent):
    def setup(self):
        self.wait_time = 0
        self.crossed = False
        self.ticks_after_cross = 0
        self.velocity = np.array([0, 0], dtype=int)
        self.lane = None
        self.pos = 0

    def axis(self):
        vx, vy = self.velocity
        if abs(vy) == 1:
            return 'NS'
        elif abs(vx) == 1:
            return 'EW'
        return None

    def propose_move(self):
        x, y = self.model.city.positions[self]
        vx, vy = self.velocity

        # Normalizar velocidad
        if (vx, vy) not in [(1,0), (-1,0), (0,1), (0,-1)]:
            self.velocity = np.array([0,1])
            vx, vy = 0, 1

        # Posición siguiente
        nx, ny = x + vx, y + vy

        # ¿Semáforo permite avanzar recto?
        axis = self.axis()
        can_move = (self.lane == self.model.current_phase[0])

        # ¿Está en celda de giro?
        will_turn = False
        if (x, y) in self.model.city.turn_cells.values():
            if random.random() < self.model.turn_prob:
                if (x, y) == self.model.city.turn_cells["NS_up_right"]:
                    self.velocity = np.array([1, 0])
                elif (x, y) == self.model.city.turn_cells["NS_down_right"]:
                    self.velocity = np.array([-1, 0])
                elif (x, y) == self.model.city.turn_cells["EW_right_down"]:
                    self.velocity = np.array([0, -1])
                elif (x, y) == self.model.city.turn_cells["EW_left_up"]:
                    self.velocity = np.array([0, 1])
                vx, vy = self.velocity
                nx, ny = x + vx, y + vy
                will_turn = True

        return (nx, ny, self.velocity, (can_move or will_turn))

# -----------------------------
# Grid de ciudad
# -----------------------------
class City(ap.Grid):
    def setup(self):
        half = self.p.size // 2
        self.turn_cells = {
            "NS_up_right":    (half+2, half-3),
            "NS_down_right":  (half-2, half+3),
            "EW_right_down":  (half-3, half-1),
            "EW_left_up":     (half+3, half+1),
        }

# -----------------------------
# Modelo de tráfico
# -----------------------------
class TrafficModel(ap.Model):
    def setup(self):
        self.size = self.p.size
        self.spawn_prob = self.p.spawn_prob
        self.turn_prob = self.p.turn_prob
        self.cycle_length = self.p.cycle_length
        self.half = self.size // 2
        self.yellow_length = self.p.get("yellow_length", 2)
        self.current_phase = 'N_GREEN'

        self.metrics = {
            "step": [],
            "cars_in_system": [],
            "cars_removed": [],
            "avg_wait_time": [],
            "total_wait_time": []
        }

        self.city = City(self, shape=(self.size, self.size))

        # Semáforo
        self.cycle_timer = 0
        self.current_phase = 'NS_GREEN'

        # Lista de autos
        self.removed_count = 0
        self.cars = ap.AgentList(self, 0, VehicleAgent)

    def spawn_vehicle(self):
        half = self.half
        side = random.choice(['N','S','E','W'])

        if side == 'N':
            pos = (half+1, 1)
            vel = np.array([0, 1], dtype=int)
        elif side == 'S':
            pos = (half-1, self.size-2)
            vel = np.array([0, -1], dtype=int)
        elif side == 'E':
            pos = (1, half-1)
            vel = np.array([1, 0], dtype=int)
        else:
            pos = (self.size-2, half+1)
            vel = np.array([-1, 0], dtype=int)

        new_car = VehicleAgent(self)
        self.cars.append(new_car)
        self.city.add_agents([new_car], positions=[pos])
        new_car.velocity = vel
        new_car.lane = side
        new_car.pos = 0

    def update_lights(self):

        self.cycle_timer += 1
        phase_duration = self.cycle_length
        yellow_duration = max(2, self.cycle_length // 3)

        if self.current_phase == 'NS_GREEN':
            if self.cycle_timer >= phase_duration:
                self.current_phase = 'NS_YELLOW'
                self.cycle_timer = 0

        elif self.current_phase == 'NS_YELLOW':
            if self.cycle_timer >= yellow_duration:   # duración corta para amarillo
                self.current_phase = 'EW_GREEN'
                self.cycle_timer = 0

        elif self.current_phase == 'EW_GREEN':
            if self.cycle_timer >= phase_duration:
                self.current_phase = 'EW_YELLOW'
                self.cycle_timer = 0

        elif self.current_phase == 'EW_YELLOW':
            if self.cycle_timer >= yellow_duration:
                self.current_phase = 'NS_GREEN'
                self.cycle_timer = 0
                

    def step(self):
        # Actualizar semáforo
        self.update_lights()

        # Spawn de autos
        if random.random() < self.spawn_prob:
            self.spawn_vehicle()

        occupied = {tuple(pos) for pos in self.city.positions.values()}
        proposals = {}
        to_remove = []

        # Propuestas de movimiento
        for car in list(self.city.agents):
            x, y = self.city.positions[car]
            nx, ny, new_vel, wants = car.propose_move()

            if nx <= 0 or nx >= self.size-1 or ny <= 0 or ny >= self.size-1:
                to_remove.append(car)
                continue

            proposals.setdefault((nx, ny), []).append((car, new_vel, wants))

        winners = {}
        for cell, lst in proposals.items():
            # Si ya hay alguien en esa celda (ocupied), no se permite que entren
            if cell in occupied:
                # Solo el que ya estaba puede quedarse
                for (car, new_vel, wants) in lst:
                    x, y = self.city.positions[car]
                    if (x, y) == cell:   # mismo lugar que antes
                        winners[car] = (cell[0], cell[1], new_vel)
                continue

    # Si la celda está libre, elegir ganador entre los que quieren moverse
            movers = [(car, new_vel) for (car, new_vel, wants) in lst if wants]
            if movers:
                car_win, v_win = random.choice(movers)
                winners[car_win] = (cell[0], cell[1], v_win)


        moved = set()
        for car, (nx, ny, new_vel) in winners.items():
            self.city.move_to(car, (nx, ny))
            car.velocity = new_vel
            car.pos += 1  
            moved.add(car)

        # Autos que no se movieron → esperar
        for car in self.city.agents:
            if car not in moved:
                car.wait_time += 1

        # Eliminar autos fuera del grid
        if to_remove:
            self.city.remove_agents(to_remove)
            self.removed_count += len(to_remove)

        #registro de las metricas
        removed_cars = len(to_remove)
        total_wait = sum(car.wait_time for car in self.city.agents)
        avg_wait = total_wait / len(self.city.agents) if self.city.agents else 0

        self.metrics["step"].append(self.t)
        self.metrics["cars_in_system"].append(len(self.city.agents))
        self.metrics["cars_removed"].append(removed_cars)
        self.metrics["avg_wait_time"].append(avg_wait)
        self.metrics["total_wait_time"].append(total_wait)

        self.t += 1

    def get_state(self):
        lights = {"N":"RED","S":"RED","E":"RED","W":"RED"}

        if self.current_phase == 'NS_GREEN':
            lights["N"] = lights["S"] = "GREEN"
        elif self.current_phase == 'NS_YELLOW':
            lights["N"] = lights["S"] = "YELLOW"
        elif self.current_phase == 'EW_GREEN':
            lights["E"] = lights["W"] = "GREEN"
        elif self.current_phase == 'EW_YELLOW':
            lights["E"] = lights["W"] = "YELLOW"

        cars_data = []
        for i, car in enumerate(self.city.agents):
            x, y = self.city.positions[car]
            cars_data.append({
                "id": i,
                "lane": car.lane,
                "pos": car.pos,
                "wait": car.wait_time,
                "x": int(x),
                "y": int(y)
            })

        return {"lights": lights, "cars": cars_data}
        

# -----------------------------
# Servidor TCP
# -----------------------------
def run_server():
    params = {"steps": 80, "size": 21, "spawn_prob": 0.3, "cycle_length": 5, "yellow_length": 2, "turn_prob": 0.3}
    model = TrafficModel(params)
    model.setup()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 5050))
    server.listen(1)
    print("Esperando conexión de Unity...")
    conn, addr = server.accept()
    print(f"Conectado con {addr}")

    for step in range(params["steps"]):
        model.step()
        data = model.get_state()
        data["step"] = step
        msg = json.dumps(data) + "\n"
        conn.sendall(msg.encode("utf-8"))
        print("Enviado:", msg.strip())
        time.sleep(0.5)

    conn.close()
    return model

if __name__ == "__main__":
    model = run_server()

#grafica de las metricas despues de la simulacion
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.plot(model.metrics["step"], model.metrics["cars_in_system"], label="Autos en sistema")
plt.xlabel("Step")
plt.ylabel("Cantidad de autos")
plt.legend()

plt.subplot(2,2,3)
plt.plot(model.metrics["step"], model.metrics["avg_wait_time"], label="Tiempo de espera promedio", color='green')
plt.xlabel("Step")
plt.ylabel("Tiempo de espera")
plt.legend()

plt.subplot(2,2,4)
plt.plot(model.metrics["step"], model.metrics["total_wait_time"], label="Tiempo de espera total", color='red')
plt.xlabel("Step")
plt.ylabel("Tiempo de espera")
plt.legend()

plt.tight_layout()
plt.show()
