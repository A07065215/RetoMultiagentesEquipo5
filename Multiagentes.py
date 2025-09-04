from matplotlib import animation
from matplotlib.patches import Rectangle
import numpy as np
import agentpy as ap
import matplotlib.pyplot as plt
from IPython.display import HTML
import random

class VehicleAgent(ap.Agent):
    def setup(self):
        "Auto con velocidad unitaria en un grid discreto"
        self.wait_time = 0
        self.city = self.model.city
        self.crossed = False #para cuando ya crucen la interseccion
        self.ticks_after_cross = 0 #pasos desde que haya cruzado
        
    def axis(self):
        #Eje de avance
        vx, vy = int(self.velocity[0]), int(self.velocity[1])
        if abs(vy) == 1 and vx == 0:
            return 'NS'
        elif abs(vx) == 1 and vy == 0:
            return 'EO'
        return None
    
    def propose_move(self):
        "Devuelve (newX, newY, new_vel, can_advance_flag)"
        #Posicion actual
        pos = self.city.positions[self]
        if pos is None:
            #aun no esta en el grid
            return (0, 0, np.array([0, 0], dtype=int), False)
        
        x, y = int(pos[0]), int(pos[1])

        #velocidad actual
        v = getattr(self, "velocity", None)
        if v is None or len(v) != 2:
            self.velocity = np.array([0, 1], dtype=int)
            v = self.velocity
        vx, vy = int(v[0]), int(v[1])

        #hay que normalizar los vectores unitarios en cad eje 
        if(vx, vy) not in [(1,0), (-1,0), (0,1), (0,-1)]:
            if abs(vx) >= abs(vy):
                vx = 1 if vx >= 0 else -1
                vy = 0
            else:
                vy = 1 if vy >= 0 else -1
                vx = 0
            self.velocity = np.array([vx, vy], dtype=int)
        
        half = self.model.half

        if vx == 0 and abs(vy) == 1:      # NS
            # Si sube, debe estar en ns_up_cols; si baja, en ns_down_cols
            allowed = self.model.ns_up_cols if vy > 0 else self.model.ns_down_cols
            if x not in allowed:
                # ajusta x a la columna de carril más cercana del conjunto permitido
                x = min(allowed, key=lambda c: abs(c - x))
                newX = x  # posición corregida en x
            # prohibir camellón central
            if x == half:
                x = half + (1 if vy > 0 else -1)
            # destino recto
            newX, newY = x, y + vy

        elif vy == 0 and abs(vx) == 1:    # EO
            # Si va a la derecha, debe estar en eo_right_rows; si a la izquierda, en eo_left_rows
            allowed = self.model.eo_right_rows if vx > 0 else self.model.eo_left_rows
            if y not in allowed:
                y = min(allowed, key=lambda r: abs(r - y))
                newY = y  # posición corregida en y
            # prohibir camellón central
            if y == half:
                y = half + (1 if vx > 0 else -1)
            # destino recto
            newX, newY = x + vx, y
        
        new_v = np.array([vx, vy], dtype=int)

        #determinar el eje
        axis = 'NS' if (vx == 0 and abs(vy) == 1) else ('EO' if (vy == 0 and abs(vx) == 1)else None)

        #ciclo del semaforo
        can_move_straight = ((axis == 'NS' and self.model.current_cycle == 0) or
                             (axis == 'EO' and self.model.current_cycle == 1))
        
        new_v = np.array([vx, vy], dtype=int)
        newX, newY = x, y

        #ver si se encuentra en una celda de giro
        will_turn = False
        half = self.model.half
        turnCells = self.city.turn_cells  # alias corto

        #Esta en alguna de las 4 celdas de giro?
        at_ns_up_right   = (int(x) == tc["NS_up_right"][0]   and int(y) == tc["NS_up_right"][1]   and vy == 1)
        at_ns_down_right = (int(x) == tc["NS_down_right"][0] and int(y) == tc["NS_down_right"][1] and vy == -1)
        at_eo_right_down = (int(x) == tc["EO_right_down"][0] and int(y) == tc["EO_right_down"][1] and vx == 1)
        at_eo_left_up    = (int(x) == tc["EO_left_up"][0]    and int(y) == tc["EO_left_up"][1]    and vx == -1)

        if self.model.turn_prob > 0 and (at_ns_up_right or at_ns_down_right or at_eo_right_down or at_eo_left_up):
            if random.random() < self.model.turn_prob:
                # Giro a la derecha según el sentido de aproximación
                if at_ns_up_right:
                    #gira a ESTE
                    new_v = np.array([1, 0], dtype=int)
                    newX, newY = x + 1, y
                elif at_ns_down_right:
                    #gira a OESTE
                    new_v = np.array([-1, 0], dtype=int)
                    newX, newY = x - 1, y
                elif at_eo_right_down:
                    #gira a SUR
                    new_v = np.array([0, -1], dtype=int)
                    newX, newY = x, y - 1
                elif at_eo_left_up:
                    #gira a NORTE
                    new_v = np.array([0, 1], dtype=int)
                    newX, newY = x, y + 1
                will_turn = True

        #se permite el movimiento si el su eje tiene el semaforo en verde o si gira y esta permitido
        if can_move_straight or will_turn:
            if will_turn:
                newX, newY = int(newX), int(newY)
            else:
                newX = int(x+int(new_v[0]))
                newY = int(y+int(new_v[1]))
        else:
            newX, newY = int(x), int(y)
        return (newX, newY, new_v, (can_move_straight or will_turn))
            


class City(ap.Grid):
    "Entorno de las celdas con los puntos de giros"
    def setup(self):
        size = self.p.size
        half = size // 2
        
        self.turn_cells = {
            "NS_up_right":    np.array([half+2, half-3]),  
            "NS_down_right":  np.array([half-2, half+3]),  
            "EO_right_down":  np.array([half-3, half-1]),  
            "EO_left_up":     np.array([half+3, half+1]),  
        }

        self.turnright_cell = self.turn_cells["NS_up_right"]
        self.turnleft_cell  = self.turn_cells["EO_right_down"] 

        stops = getattr(self.p, 'stops', [])
        self.bus_stops = dict(zip(stops, [0]*len(stops)))


        def update(self):
            for stop in self.bus_stops.keys():
                if random.random() < 0.1:
                    self.bus_stops[stop] = random.randint(0, 10)


class TrafficModel(ap.Model):
    def setup(self):
        #Parametros del modelo
        self.size = int(self.p.size) #Tamaño del grid
        self.lanes_offsets = [-2, -1, 1, 2] #Numero de carriles por direccion
        self.spawn_prob = float(self.p.spawn_prob) #Probabilidad de que aparezca un auto en cada carril
        self.cycle_length = int(self.p.cycle_length) #Duracion del ciclo del semaforo
        self.turn_prob = float(self.p.turn_prob) #Probabilidad de que un auto gire en la interseccion
        
        #4 carriles por lado (2 por sentido)
        self.half = self.size // 2
        
        #norte a sur
        self.ns_down_cols = [self.half-2, self.half-1]
        self.ns_up_cols = [self.half+1, self.half+2]
        self.ns_all_cols = set(self.ns_down_cols + self.ns_up_cols)

        #este a oeste
        self.eo_right_rows = [self.half-2, self.half-1]
        self.eo_left_rows = [self.half+1, self.half+2]
        self.eo_all_rows = set(self.eo_left_rows + self.eo_right_rows)

        #Entorno grid
        self.city = City(self, shape=(self.size, self.size))

        #Semaforos centralizados en los ejes
        self.current_cycle = 0
        self.cycle_timer = 0

        #Para animar las metricas
        self.positions = [] #lista por paso de las posiciones de los autos
        self.removed_count = 0 #autos que ya salieron del grid

    #Spawn de los carros segun los carriles
    def spawn_vehicles(self):
        half = self.size // 2

        #Elige aleatoriamente el eje y el sentido de cad auto
        axis_choice = random.choice(['NS', 'EO'])
        lane_offset = random.choice(self.lanes_offsets)

        if axis_choice == 'NS':
            #carriles paralelos al eje y, manejando x alrededor del centro
            x_lane = half + lane_offset
            if random.random() < 0.5: #de sur a norte y continua creciendo
                x = random.choice(self.ns_up_cols)
                pos = np.array([x, 1])
                vel = np.array([0, 1])
            else: #de norte a sur decrece
                x = random.choice(self.ns_up_cols)
                pos = np.array([x, self.size-2])
                vel = np.array([0, -1])
        else: #Eje EO
            if random.random() < 0.5:
                #entra por izquiera y va hacia la derecha
                y = random.choice(self.eo_right_rows)
                pos = np.array([1, y])
                vel = np.array([1, 0])
            else:
                #entra por derecha y va hacia la izquierda
                y = random.choice(self.eo_left_rows)
                pos = np.array([self.size-2, y])
                vel = np.array([-1, 0])
        
        cars = ap.AgentList(self, 1, VehicleAgent)
        self.city.add_agents(cars, positions =[pos])
        cars = cars[0]
        cars.velocity = np.array(vel, dtype=int)

    def step(self):
        #Ciclo del semaforo
        if self.cycle_timer == 0:
            pass
        self.cycle_timer += 1
        if self.cycle_timer >= self.cycle_length:
            self.cycle_timer = 0
            self.current_cycle = 1 - self.current_cycle #Alterna entre 0 y 1

        #Spawn de autos en los bordes
        if random.random() < self.spawn_prob:
            self.spawn_vehicles()

        #Proponer movimientos
        #ocupacion actual del grid 
        occupied = {tuple(pos) for pos in self.city.positions.values()}

        proposals = {}
        stayer = set()
        to_remove = []

        for car in list(self.city.agents):
            x, y = self.city.positions[car]

            #checar si sale del grid despues de su movimiento
            pm = car.propose_move()
            if not isinstance(pm, tuple) or len(pm) != 4:
                #si sucede algo raro no se propone movimiento
                newX, newY, new_vel, wants_to_move = (self.city.positions[car][0],
                                                  self.city.positions[car][1],
                                                  getattr(car, "velocity", np.array([0, 0], dtype=int)),
                                                  False)
            else:
                newX, newY, new_vel, wants_to_move = pm
            
            newX, newY = int(newX), int(newY)

            #si su destino queda fuera del grid se elimina
            if newX <= 0 or newX >= self.size - 1 or newY <= 0 or newY >= self.size - 1:
                #si ya esta en el borde se elimina
                if (x <= 0 or x >= self.size - 1 or y <= 0 or y >= self.size -1):
                    to_remove.append(car)
                    continue
                pass

            #registrar la propuesta
            key = (newX, newY)
            proposals.setdefault(key, []).append((car, new_vel, wants_to_move))

            #si en la propuesta el auto se queda, lo marcamos como stayer
            if (newX, newY) == (x, y):
                stayer.add(car)
        
        #Resolver conflictos por cada celda destino
        winners = {}
        losers = set()

        for cell, lst in proposals.items():
            #Si ya esta ocupada por alguien que se queda, no se mueve solo puede conservarla el que ya esta ahi
            if cell in occupied:
                for (car, new_vel, _) in lst:
                    x, y = self.city.positions[car]
                    if (x, y) == cell:
                        winners[car] = (cell[0], cell[1], new_vel) #se queda
                    else:
                        losers.add(car)
                continue

            #si varios carros quieren entrar, se prioriza a los que realmente se estn moviendo y cuyo eje este en verde
            movers = [(car, new_vel) for (car, new_vel, wants) in lst
                      if wants and tuple(self.city.positions[car]) != cell]
            
            if movers:
                car_win, v_win = random.choice(movers)
                winners[car_win] = (cell[0], cell[1], v_win)
                for (car, _) in movers:
                    if car is not car_win:
                        losers.add(car)

                #en caso de que existan stayers en la celda, se marcan como perdedores por seguridad
                for (car, new_vel, wants) in lst:
                    if (car not in winners) and ((self.city.positions[car][0], self.city.positions[car][1]) == cell):
                        losers.add(car)
            else:
                for (car, new_vel, _) in lst:
                    x, y = self.city.positions[car]
                    if (x, y) == cell:
                        winners[car] = (cell[0], cell[1], new_vel)
                    else:
                        losers.add(car)

        #Aplicar movimientos y contabilizar tiempos de espera
        moved = set()
        for car, (newX, newY, new_vel) in winners.items():
            x, y = self.city.positions[car]
            dx, dy = int(newX - x), int(newY - y)
            if dx != 0 or dy != 0:
                #mover y actualizar su velocidad
                self.city.move_by(car, np.array([dx, dy], dtype=int))
                car.velocity = new_vel
                moved.add(car)
            else:
                #si se queda en su lugar no suma la espera
                car.velocity = new_vel

        #perdida del turno de espera
        for car in self.city.agents:
            if car not in moved:
                car.wait_time += 1

        #ciclo de vida marcando el cruce y eliminar tras un lapso de tiempo
        half = self.size // 2
        DESPAWN_AFTER = 10 #pasos tras cruzar o girar

        to_remove_extra = []

        for car in list(self.city.agents):
            #pos actual
            x, y = map(int, self.city.positions[car])
            vx, vy = map(int, getattr(car, "velocity", np.array([0,0])))

            is_turn_cell = any(x == c[0] and y == c[1] for c in self.city.turn_cells.values())
            
            #marcar que ya paso el cruce cuando se encuntra fuera de la linea central
            if not car.crossed:
                if vx != 0:
                    #si se mueve en este-oeste considerar que ya cruzo al pasar por la columna central
                    if (vx > 0 and y > half) or (vy < 0 and y < half):
                        car.crossed = True
                        car.ticks_after_cross = 0
                elif vy != 0:
                    if(vy > 0 and y > self.half) or (vy < 0 and y < self.half):
                        car.crossed = True
                        car.ticks_after_cross = 0
                
                if is_turn_cell:
                    car.crossed = True
                    car.ticks_after_cross = 0

            
            #si ya cruzo se incrementa el contador y lo remueve despues de los pasos establecidos
            if car.crossed:
                car.ticks_after_cross += 1
                if car.ticks_after_cross >= DESPAWN_AFTER:
                    to_remove_extra.append(car)
            
        if to_remove_extra:
            self.city.remove_agents(to_remove_extra)
            self.removed_count += len(to_remove_extra)
            
        #eliminar los autos que salieron del grid
        if to_remove:
            self.city.remove_agents(to_remove)
            self.removed_count += len(to_remove)
        
        #actualizar el entorno
        #self.city.update()

        #guardar con snapshot las posiciones para la animacion
        snap = []
        for car in self.city.agents:
            x, y = self.city.positions[car]
            axis = car.axis()
            snap.append((int(x), int(y), 0 if axis == 'NS' else 1))
        self.positions.append(snap)


#parametros del modelo 
parameters = {
    'steps': 100,
    'size': 21,
    'lanes': 4,
    'spawn_prob': 0.3,
    'cycle_length': 8,
    'turn_prob': 0.3,
    'stops': [(2, 5)]
}

model = TrafficModel(parameters)
results = model.run()

if not hasattr(model, "positions") or len(model.positions) == 0:
    raise RuntimeError("No hay frames en model.positions. Asegúrate de llenar model.positions en cada step().")

# Funciones de dibujo
fig, ax = plt.subplots(figsize=(7, 7))
half = model.size // 2

def _draw_static_background():
    ax.clear()
    ax.set_title("Tráfico con semáforos")

    half = model.half
    road_half_width = 4  # 4 celdas por lado = 8 total

    #calle
    ax.add_patch(Rectangle((0, half-road_half_width), model.size-1, 2*road_half_width,
                           facecolor="#e6e6e6", edgecolor="#e6e6e6"))
    ax.add_patch(Rectangle((half-road_half_width, 0), 2*road_half_width, model.size-1,
                           facecolor="#e6e6e6", edgecolor="#e6e6e6"))

    # Camellón vertical y horizontal (una celda de ancho)
    ax.add_patch(Rectangle((half-0.5, 0), 1, model.size-1, facecolor="#c5c5c5", edgecolor="#c5c5c5"))
    ax.add_patch(Rectangle((0, half-0.5), model.size-1, 1, facecolor="#c5c5c5", edgecolor="#c5c5c5"))

    # Líneas punteadas de carril: ±1 y ±2
    for off in [-2, -1, 1, 2]:
        # EO (horizontales)
        ax.plot([0, model.size-1], [half+off, half+off], linestyle="--", linewidth=0.8, color="0.55", dashes=(4,4))
        # NS (verticales)
        ax.plot([half+off, half+off], [0, model.size-1], linestyle="--", linewidth=0.8, color="0.55", dashes=(4,4))

    # Celdas de giro (visuales)
    # Celdas de giro (4 verdes)
    for cell in model.city.turn_cells.values():
        ax.scatter(cell[0], cell[1], s=90, marker="s", c="#66ff66", edgecolors="k", linewidths=0.5)

    # Paradas de bus
    if hasattr(model.city, "bus_stops") and model.city.bus_stops:
        sxy = np.array(list(model.city.bus_stops.keys())).T
        ax.scatter(*sxy, s=90, marker="8", c="red", edgecolors="k", linewidths=0.5)

    ax.set_xlim(0, model.size-1)
    ax.set_ylim(0, model.size-1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

def _phase_text_for_t(t, cycle_length):
    current_cycle = (t // cycle_length) % 2
    return "NS = VERDE  |  EO = ROJO" if current_cycle == 0 else "EO = VERDE  |  NS = ROJO"

def _scatter_pairs(pairs, marker, label):
    if pairs:
        arr = np.array(pairs).T
        ax.scatter(*arr, s=60, marker=marker, c="black", label=label)

def _update(frame_idx):
    _draw_static_background()
    ax.set_title(f"Tráfico con semáforos — paso t={frame_idx}")

    #Texto de fase reconstruida a partir de t (coincide si se alterna el cycle_length del semaforo)
    phase_txt = _phase_text_for_t(frame_idx, model.cycle_length)
    ax.text(0.5, 1.02, phase_txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color="green")

    #Datos guardados: cada frame es [(x,y,eje), ...] con eje: 0=NS, 1=EO
    data = model.positions[frame_idx]
    ns_up, ns_down, eo_right, eo_left = [], [], [], []

    #Heurística de sentido usando frame previo (si existe)
    prev = model.positions[frame_idx-1] if frame_idx > 0 else None
    prev_map = {(d[0], d[1]): True for d in prev} if prev else {}

    for (x, y, eje) in data:
        x, y, eje = int(x), int(y), int(eje)
        if eje == 0:  # NS
            if prev:
                if (x, y-1) in prev_map:  #de abajo para arriba
                    ns_up.append((x, y))
                elif (x, y+1) in prev_map:  #de arriba para abajo
                    ns_down.append((x, y))
                else:
                    ns_up.append((x, y))    # fallback
            else:
                ns_up.append((x, y))
        else:         # EO
            if prev:
                if (x-1, y) in prev_map:    # viene de izquierda a derecha
                    eo_right.append((x, y))
                elif (x+1, y) in prev_map:  # viene derecha a izquierda
                    eo_left.append((x, y))
                else:
                    eo_right.append((x, y)) # fallback
            else:
                eo_right.append((x, y))

    _scatter_pairs(ns_up,    "^", "NS ↑")
    _scatter_pairs(ns_down,  "v", "NS ↓")
    _scatter_pairs(eo_right, ">", "EO →")
    _scatter_pairs(eo_left,  "<", "EO ←")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=9, frameon=True)
    return []

ani = animation.FuncAnimation(fig, _update, frames=len(model.positions), interval=180, blit=False)

plt.show()