import numpy as np
import agentpy as ap
import matplotlib.pyplot as plt
from IPython.display import HTML

class VehicleAgent(ap.Agent):
    def setup(self):
        self.wait_time = 0
        # Asignar posición inicial según dirección
        # 0: norte-sur, 1: sur-norte, 2: este-oeste, 3: oeste-este
        if hasattr(self, 'direction'):
            if self.direction == 0:
                self.x, self.y = 5, 10
            elif self.direction == 1:
                self.x, self.y = 5, 0
            elif self.direction == 2:
                self.x, self.y = 10, 5
            elif self.direction == 3:
                self.x, self.y = 0, 5
        else:
            self.x, self.y = 5, 5

    def step(self):
        # Movimiento simple: avanza si el semáforo está en verde, si no espera
        # Determinar semáforo correspondiente
        light = self.model.traffic_lights[self.direction]
        if light.state == 'green':
            if self.direction == 0 and self.y > 0:
                self.y -= 1
            elif self.direction == 1 and self.y < 10:
                self.y += 1
            elif self.direction == 2 and self.x > 0:
                self.x -= 1
            elif self.direction == 3 and self.x < 10:
                self.x += 1
        else:
            self.wait_time += 1

    def move(self):
        pass  # No se usa en esta versión

class TrafficLightAgent(ap.Agent):
    def setup(self):
        self.state = 'red'
        # El estado será controlado por el modelo central

class TrafficModel(ap.Model):
    def setup(self):
        #Parametros ajustables del modelo
        self.light_cycle = {0: 0, 1: 0, 2: 0, 3: 0} #Duracion de los semaforos
        self.traffic_lights = ap.AgentList(self, 4, TrafficLightAgent)
        self.vehicles = ap.AgentList(self, 10, VehicleAgent)

        for i, v in enumerate(self.vehicles):
            v.direction = i % 4
        #Asignacion de la direccion para los vehiculos
        for v in self.vehicles:
            if v.direction == 0:
                v.x, v.y = 5, 10
            elif v.direction == 1:
                v.x, v.y = 5, 0
            elif v.direction == 2:
                v.x, v.y = 10, 5
            elif v.direction == 3:
                v.x, v.y = 0, 5
    # Guardar posiciones para animación
        self.positions = []
    # Controlador centralizado de semáforos
        self.cycle_length = 6  # duración de cada ciclo de semáforo
        self.current_cycle = 0  # 0: norte-sur, 1: este-oeste
        self.cycle_timer = 0
    
    def step(self):

        occupied = {(v.x, v.y) for v in self.vehicles}

        proposals = []
        for v in self.vehicles:
            nx, ny = v.x, v.y
            light = self.traffic_lights[v.direction]
            if light.state == 'green':
                if v.direction == 0 and v.y > 0: 
                    ny = v.y - 1
                elif v.direction == 1 and v.y < 10: 
                    ny = v.y + 1
                elif v.direction == 2 and v.x > 0: 
                    nx = v.x - 1
                elif v.direction == 3 and v.x < 10: 
                    nx = v.x + 1
            proposals.append((v, (nx, ny)))

        from collections import defaultdict
        bucket = defaultdict(list)
        for v, (nx,ny) in proposals:
            bucket[(nx,ny)].append(v)

        approved = set()
        for cell, vv in bucket.items():
            if cell in occupied:
            #el carro que ya se encuentre y los demas se esperan
                for v in vv:
                    if (v.x,v.y) == cell:
                        approved.add((v, cell))
                    continue
            movers = [v for v in vv if (v.x,v.y) != cell and self.traffic_lights[v.direction].state == 'green']
            if movers:
                #elige uno o aplica prioridad
                winner = self.random.choice(movers)
                approved.add((winner, cell))
            else:
                #si todos se quedan, se aprueban en su celdas
                for v in vv:
                    if (v.x, v.y) == cell:
                        approved.add((v, cell))
                        
        moved = set()
        for v, (nx, ny) in approved:
            if (nx, ny) != (v.x, v.y):
                v.x, v.y = nx, ny
                moved.add(v)

        for v in self.vehicles:
            if v not in moved:
                #si no se movio aumenta el tiempo de espera
                v.wait_time += 1

        # Controlador centralizado de semáforos
        if self.cycle_timer == 0:
            if self.current_cycle == 0:
                # Verde para norte-sur (0 y 1), rojo para este-oeste (2 y 3)
                self.traffic_lights[0].state = 'green'
                self.traffic_lights[1].state = 'green'
                self.traffic_lights[2].state = 'red'
                self.traffic_lights[3].state = 'red'
            else:
                # Verde para este-oeste (2 y 3), rojo para norte-sur (0 y 1)
                self.traffic_lights[0].state = 'red'
                self.traffic_lights[1].state = 'red'
                self.traffic_lights[2].state = 'green'
                self.traffic_lights[3].state = 'green'
        self.cycle_timer += 1
        if self.cycle_timer >= self.cycle_length:
            self.cycle_timer = 0
            self.current_cycle = 1 - self.current_cycle

        # Los semáforos ya no cambian solos
        # self.traffic_lights.step()  # No llamar step de semáforos
        self.vehicles.step()
        # Guardar posiciones de todos los vehículos
        self.positions.append([(v.x, v.y, v.direction) for v in self.vehicles])
        ns_wait = sum(v.wait_time for v in self.vehicles if v.direction in (0,1))
        eo_wait = sum(v.wait_time for v in self.vehicles if v.direction in (2,3))
        self.next_ns_bonus = 2 if ns_wait > eo_wait else 0
parameters = {
    'steps': 20,
}

model = TrafficModel(parameters)
results = model.run()

# Animación con matplotlib
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Simulación de cruce de 4 semáforos')

# Dibujar líneas del cruce
ax.plot([0, 10], [5, 5], color='gray', linewidth=2)
ax.plot([5, 5], [0, 10], color='gray', linewidth=2)

colors = ['red', 'blue', 'green', 'orange']
scat = ax.scatter([], [], s=100)

def update(frame):
    data = model.positions[frame]
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    cs = [colors[d[2]] for d in data]
    scat.set_offsets(np.c_[xs, ys])
    scat.set_color(cs)
    ax.set_title(f'Simulación de cruce - Paso {frame+1}')
    return scat,

ani = animation.FuncAnimation(fig, update, frames=len(model.positions), interval=500, blit=True)
plt.show()

# Calcular tiempo de espera total y promedio
total_wait_time = sum(v.wait_time for v in model.vehicles)
avg_wait_time = total_wait_time / len(model.vehicles)

# Contar vehículos por dirección
from collections import Counter
direction_counts = Counter(v.direction for v in model.vehicles)
max_dir = max(direction_counts, key=direction_counts.get)

print(f"Tiempo total de espera: {total_wait_time}")
print(f"Tiempo promedio de espera por vehiculo: {avg_wait_time:.2f}")
print("Trafico por direccion:", direction_counts)
print(f"La direccion con mas trafico fue: {max_dir} (con {direction_counts[max_dir]} vehiculos)")