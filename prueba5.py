import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
from typing import List, Tuple, Union, Optional
import json
import csv

class TSPGeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 100,
                 max_iterations: int = 1000,
                 survivors_pct: float = 0.2,
                 crossover_pct: float = 0.6,
                 mutation_pct: float = 0.2,
                 mutation_rate: float = 0.1,
                 elite_size: int = 5):
        """
        Algoritmo Genético para TSP
        
        Args:
            population_size: Tamaño de la población
            max_iterations: Número máximo de iteraciones
            survivors_pct: Porcentaje de sobrevivientes
            crossover_pct: Porcentaje de población creada por cruce
            mutation_pct: Porcentaje de población creada por mutación
            mutation_rate: Probabilidad de mutación por gen
            elite_size: Número de mejores individuos que siempre sobreviven
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.survivors_count = int(population_size * survivors_pct)
        self.crossover_count = int(population_size * crossover_pct)
        self.mutation_count = int(population_size * mutation_pct)
        self.mutation_rate = mutation_rate
        self.elite_size = min(elite_size, self.survivors_count)
        
        # Ajustar para que sume exactamente population_size
        total = self.survivors_count + self.crossover_count + self.mutation_count
        if total != population_size:
            diff = population_size - total
            self.crossover_count += diff
        
        self.cities = None
        self.distance_matrix = None
        self.num_cities = 0
        self.population = []
        self.fitness_history = []
        self.best_route_history = []
        self.diversity_history = []
        
    def load_coordinates_from_file(self, filename: str):
        """Cargar coordenadas desde archivo CSV o JSON"""
        try:
            if filename.endswith('.json'):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.cities = np.array(data)
            elif filename.endswith('.csv'):
                self.cities = np.loadtxt(filename, delimiter=',')
            else:
                # Asumir formato de texto simple: x,y por línea
                self.cities = np.loadtxt(filename, delimiter=',')
                
            self.num_cities = len(self.cities)
            self._calculate_distance_matrix_from_coords()
            print(f"Cargadas {self.num_cities} ciudades desde {filename}")
        except Exception as e:
            print(f"Error cargando coordenadas: {e}")
            
    def load_coordinates_from_list(self, coordinates: List[Tuple[float, float]]):
        """Cargar coordenadas desde una lista"""
        self.cities = np.array(coordinates)
        self.num_cities = len(self.cities)
        self._calculate_distance_matrix_from_coords()
        print(f"Cargadas {self.num_cities} ciudades desde lista")
        
    def load_distance_matrix_from_file(self, filename: str):
        """Cargar matriz de distancias desde archivo"""
        try:
            if filename.endswith('.json'):
                with open(filename, 'r') as f:
                    self.distance_matrix = np.array(json.load(f))
            else:
                self.distance_matrix = np.loadtxt(filename, delimiter=',')
                
            self.num_cities = len(self.distance_matrix)
            # Generar coordenadas ficticias para visualización
            self._generate_dummy_coordinates()
            print(f"Cargada matriz de distancias {self.num_cities}x{self.num_cities}")
        except Exception as e:
            print(f"Error cargando matriz de distancias: {e}")
            
    def load_distance_matrix_from_array(self, matrix: np.ndarray):
        """Cargar matriz de distancias desde array numpy"""
        self.distance_matrix = matrix
        self.num_cities = len(matrix)
        self._generate_dummy_coordinates()
        print(f"Cargada matriz de distancias {self.num_cities}x{self.num_cities}")
        
    def _calculate_distance_matrix_from_coords(self):
        """Calcular matriz de distancias desde coordenadas"""
        self.distance_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    dist = np.sqrt((self.cities[i][0] - self.cities[j][0])**2 + 
                                 (self.cities[i][1] - self.cities[j][1])**2)
                    self.distance_matrix[i][j] = dist
                    
    def _generate_dummy_coordinates(self):
        """Generar coordenadas ficticias para visualización cuando solo hay matriz"""
        # Usar MDS (Multidimensional Scaling) simplificado
        angles = np.linspace(0, 2*np.pi, self.num_cities, endpoint=False)
        radius = 10
        self.cities = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ])
        
    def _initialize_population(self):
        """Inicializar población con diversidad garantizada"""
        self.population = []
        
        # Crear algunos individuos completamente aleatorios
        for _ in range(self.population_size // 2):
            route = list(range(self.num_cities))
            random.shuffle(route)
            self.population.append(route)
            
        # Crear individuos con heurísticas simples para diversidad
        for _ in range(self.population_size - len(self.population)):
            route = self._nearest_neighbor_heuristic(random.randint(0, self.num_cities-1))
            # Aplicar mutaciones aleatorias
            for _ in range(random.randint(1, 5)):
                route = self._mutate_2opt(route.copy())
            self.population.append(route)
            
    def _nearest_neighbor_heuristic(self, start_city: int) -> List[int]:
        """Heurística del vecino más cercano"""
        route = [start_city]
        remaining = set(range(self.num_cities)) - {start_city}
        
        current = start_city
        while remaining:
            nearest = min(remaining, key=lambda x: self.distance_matrix[current][x])
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest
            
        return route
        
    def _calculate_fitness(self, route: List[int]) -> float:
        """Calcular fitness (inverso de la distancia total)"""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += self.distance_matrix[from_city][to_city]
        return 1 / (total_distance + 1e-8)  # Evitar división por cero
        
    def _calculate_distance(self, route: List[int]) -> float:
        """Calcular distancia total de una ruta"""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance
        
    def _tournament_selection(self, population: List[List[int]], k: int = 5) -> List[int]:
        """Selección por torneo"""
        tournament = random.sample(population, min(k, len(population)))
        return max(tournament, key=self._calculate_fitness)
        
    def _roulette_selection(self, population: List[List[int]]) -> List[int]:
        """Selección por ruleta"""
        fitness_scores = [self._calculate_fitness(route) for route in population]
        total_fitness = sum(fitness_scores)
        
        if total_fitness == 0:
            return random.choice(population)
            
        probabilities = [f/total_fitness for f in fitness_scores]
        return np.random.choice(population, p=probabilities)
        
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Operador de cruce por orden (OX)"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copiar segmento del primer padre
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Llenar resto con orden del segundo padre
        self._fill_child(child1, parent2, start, end)
        self._fill_child(child2, parent1, start, end)
        
        return child1, child2
        
    def _fill_child(self, child: List[int], parent: List[int], start: int, end: int):
        """Llenar hijo con genes faltantes manteniendo orden"""
        child_set = set(child[start:end])
        parent_filtered = [gene for gene in parent if gene not in child_set]
        
        # Llenar posiciones antes del segmento
        idx = 0
        for i in range(start):
            child[i] = parent_filtered[idx]
            idx += 1
            
        # Llenar posiciones después del segmento
        for i in range(end, len(child)):
            child[i] = parent_filtered[idx]
            idx += 1
            
    def _pmx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Cruce por mapeo parcial (PMX)"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Intercambiar segmentos
        child1[start:end], child2[start:end] = parent2[start:end], parent1[start:end]
        
        # Resolver conflictos
        self._resolve_pmx_conflicts(child1, parent1, parent2, start, end)
        self._resolve_pmx_conflicts(child2, parent2, parent1, start, end)
        
        return child1, child2
        
    def _resolve_pmx_conflicts(self, child: List[int], parent1: List[int], parent2: List[int], 
                              start: int, end: int):
        """Resolver conflictos en PMX"""
        for i in range(len(child)):
            if i < start or i >= end:
                if child[i] in child[start:end]:
                    # Buscar reemplazo
                    current = child[i]
                    while current in child[start:end]:
                        idx = parent2[start:end].index(current)
                        current = parent1[start + idx]
                    child[i] = current
                    
    def _mutate_2opt(self, route: List[int]) -> List[int]:
        """Mutación 2-opt"""
        if len(route) < 4:
            return route
            
        route = route.copy()
        i, j = sorted(random.sample(range(len(route)), 2))
        if j - i < 2:
            return route
            
        # Invertir segmento
        route[i:j+1] = route[i:j+1][::-1]
        return route
        
    def _mutate_swap(self, route: List[int]) -> List[int]:
        """Mutación por intercambio"""
        route = route.copy()
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
        return route
        
    def _mutate_scramble(self, route: List[int]) -> List[int]:
        """Mutación por mezcla de segmento"""
        route = route.copy()
        start = random.randint(0, len(route) - 3)
        end = random.randint(start + 2, len(route))
        segment = route[start:end]
        random.shuffle(segment)
        route[start:end] = segment
        return route
        
    def _calculate_diversity(self) -> float:
        """Calcular diversidad de la población"""
        if len(self.population) < 2:
            return 0.0
            
        total_distance = 0
        comparisons = 0
        
        for i in range(min(20, len(self.population))):  # Muestreo para eficiencia
            for j in range(i + 1, min(20, len(self.population))):
                # Distancia de Hamming normalizada
                diff = sum(1 for a, b in zip(self.population[i], self.population[j]) if a != b)
                total_distance += diff / len(self.population[i])
                comparisons += 1
                
        return total_distance / comparisons if comparisons > 0 else 0.0
        
    def _maintain_diversity(self):
        """Mantener diversidad en la población"""
        diversity = self._calculate_diversity()
        
        if diversity < 0.1:  # Umbral de diversidad mínima
            # Reemplazar individuos similares con nuevos aleatorios
            num_replace = self.population_size // 4
            fitness_scores = [(i, self._calculate_fitness(route)) 
                            for i, route in enumerate(self.population)]
            fitness_scores.sort(key=lambda x: x[1])
            
            # Mantener los mejores, reemplazar algunos de los peores
            for i in range(num_replace):
                idx = fitness_scores[i][0]
                new_route = list(range(self.num_cities))
                random.shuffle(new_route)
                # Aplicar algunas mutaciones
                for _ in range(random.randint(2, 5)):
                    if random.random() < 0.5:
                        new_route = self._mutate_2opt(new_route)
                    else:
                        new_route = self._mutate_swap(new_route)
                self.population[idx] = new_route
                
    def solve(self, verbose: bool = True) -> Tuple[List[int], float]:
        """Resolver TSP usando algoritmo genético"""
        if self.distance_matrix is None:
            raise ValueError("No se ha cargado información de ciudades o distancias")
            
        self._initialize_population()
        self.fitness_history = []
        self.best_route_history = []
        self.diversity_history = []
        
        for generation in range(self.max_iterations):
            # Evaluar fitness
            fitness_scores = [(route, self._calculate_fitness(route)) 
                            for route in self.population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Guardar mejor solución
            best_route = fitness_scores[0][0]
            best_distance = self._calculate_distance(best_route)
            self.fitness_history.append(best_distance)
            self.best_route_history.append(best_route.copy())
            
            # Calcular y guardar diversidad
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            if verbose and generation % 50 == 0:
                print(f"Generación {generation}: Mejor distancia = {best_distance:.2f}, "
                      f"Diversidad = {diversity:.3f}")
                      
            # Selección de sobrevivientes (élite + selección)
            new_population = []
            
            # Élite
            for i in range(self.elite_size):
                new_population.append(fitness_scores[i][0])
                
            # Sobrevivientes adicionales por selección
            for _ in range(self.survivors_count - self.elite_size):
                survivor = self._tournament_selection([r[0] for r in fitness_scores])
                new_population.append(survivor)
                
            # Cruce
            for _ in range(self.crossover_count // 2):
                parent1 = self._tournament_selection(self.population)
                parent2 = self._tournament_selection(self.population)
                
                if random.random() < 0.7:  # Probabilidad de cruce
                    if random.random() < 0.5:
                        child1, child2 = self._order_crossover(parent1, parent2)
                    else:
                        child1, child2 = self._pmx_crossover(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1.copy(), parent2.copy()])
                    
            # Mutación
            for _ in range(self.mutation_count):
                parent = random.choice(self.population)
                child = parent.copy()
                
                # Aplicar múltiples tipos de mutación
                if random.random() < self.mutation_rate:
                    mutation_type = random.choice(['2opt', 'swap', 'scramble'])
                    if mutation_type == '2opt':
                        child = self._mutate_2opt(child)
                    elif mutation_type == 'swap':
                        child = self._mutate_swap(child)
                    else:
                        child = self._mutate_scramble(child)
                        
                new_population.append(child)
                
            # Ajustar tamaño de población
            while len(new_population) < self.population_size:
                new_population.append(random.choice(new_population).copy())
            if len(new_population) > self.population_size:
                new_population = new_population[:self.population_size]
                
            self.population = new_population
            
            # Mantener diversidad
            if generation % 20 == 0:
                self._maintain_diversity()
                
        # Obtener mejor solución final
        final_fitness = [(route, self._calculate_fitness(route)) 
                        for route in self.population]
        final_fitness.sort(key=lambda x: x[1], reverse=True)
        
        best_route = final_fitness[0][0]
        best_distance = self._calculate_distance(best_route)
        
        if verbose:
            print(f"\n--- Resultado Final ---")
            print(f"Mejor ruta encontrada: {best_route}")
            print(f"Distancia total: {best_distance:.2f}")
            
        return best_route, best_distance
        
    def animate_best_route(self, save_gif: bool = False, filename: str = "tsp_best_route.gif"):
        """Crear animación del recorrido de la mejor ruta encontrada"""
        if not self.best_route_history:
            print("No hay datos de evolución para mostrar")
            return
            
        # Obtener la mejor ruta final
        best_route = self.best_route_history[-1]
        best_distance = self.fitness_history[-1]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            ax.clear()
            
            # Dibujar todas las ciudades
            ax.scatter(self.cities[:, 0], self.cities[:, 1], c='lightcoral', s=100, 
                      zorder=3, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Numerar las ciudades
            for i, (x, y) in enumerate(self.cities):
                ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold')
            
            # Dibujar el recorrido hasta el frame actual
            num_segments = len(best_route)
            segments_to_show = min(frame + 1, num_segments)
            
            # Dibujar segmentos del recorrido
            for i in range(segments_to_show):
                from_city = best_route[i]
                to_city = best_route[(i + 1) % len(best_route)]
                
                # Color más intenso para el segmento actual
                if i == segments_to_show - 1:
                    ax.plot([self.cities[from_city][0], self.cities[to_city][0]],
                           [self.cities[from_city][1], self.cities[to_city][1]], 
                           'blue', linewidth=4, alpha=0.9, zorder=2)
                else:
                    ax.plot([self.cities[from_city][0], self.cities[to_city][0]],
                           [self.cities[from_city][1], self.cities[to_city][1]], 
                           'steelblue', linewidth=2, alpha=0.6, zorder=1)
            
            # Marcar ciudad actual
            if segments_to_show > 0:
                current_city = best_route[segments_to_show - 1]
                ax.scatter(self.cities[current_city][0], self.cities[current_city][1], 
                          c='lime', s=200, marker='o', zorder=4, 
                          edgecolors='darkgreen', linewidth=2, label='Posición Actual')
            
            # Marcar ciudad de inicio
            start_city = best_route[0]
            ax.scatter(self.cities[start_city][0], self.cities[start_city][1], 
                      c='gold', s=150, marker='s', zorder=4, 
                      edgecolors='darkorange', linewidth=2, label='Inicio')
            
            ax.set_title(f'Mejor Ruta TSP - Simulación del Recorrido\n'
                        f'Distancia Total: {best_distance:.2f} | '
                        f'Segmento: {segments_to_show}/{num_segments}', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            # Ajustar límites para mejor visualización
            margin = 0.1 * max(np.ptp(self.cities[:, 0]), np.ptp(self.cities[:, 1]))
            ax.set_xlim(self.cities[:, 0].min() - margin, self.cities[:, 0].max() + margin)
            ax.set_ylim(self.cities[:, 1].min() - margin, self.cities[:, 1].max() + margin)
        
        # Crear animación (recorre cada segmento + pausa al final)
        total_frames = len(best_route) + 20  # +20 para pausa al final
        ani = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                    interval=800, repeat=True, blit=False)
        
        plt.tight_layout()
        
        if save_gif:
            try:
                ani.save(filename, writer='pillow', fps=1.5)
                print(f"Animación guardada como {filename}")
            except Exception as e:
                print(f"Error guardando animación: {e}")
                
        plt.show()
        return ani


# Ejemplo de uso y pruebas
def main():
    """Ejemplo de uso del algoritmo genético para TSP"""
    
    print("=== ALGORITMO GENÉTICO PARA TSP ===\n")
    
    # Crear y configurar algoritmo
    tsp = TSPGeneticAlgorithm(
        population_size=50,
        max_iterations=200,
        survivors_pct=0.3,
        crossover_pct=0.5,
        mutation_pct=0.2,
        mutation_rate=0.15,
        elite_size=3
    )
    
    # Intentar cargar desde archivo CSV, si no existe usar datos por defecto
    try:
        print("1. Intentando cargar coordenadas desde 'ciudades.csv'...")
        tsp.load_coordinates_from_file('ciudades.csv')
    except:
        print("   Archivo 'ciudades.csv' no encontrado. Usando datos por defecto...")
        # Datos por defecto si no existe el archivo
        default_cities = [
            (12.5, 87.3), (45.2, 23.8), (78.9, 65.4), (23.1, 45.7), (89.6, 12.3),
            (34.7, 78.9), (67.8, 34.5), (15.4, 56.2), (92.3, 89.7), (56.7, 8.9),
            (8.2, 34.6), (73.4, 91.2), (28.9, 67.8), (85.1, 45.3), (41.6, 72.1)
        ]
        tsp.load_coordinates_from_list(default_cities)
    
    # Resolver
    print("\n2. Ejecutando algoritmo genético...")
    best_route, best_distance = tsp.solve(verbose=True)
    
    # Crear simulación visual del mejor recorrido
    print("\n3. Creando simulación visual del mejor recorrido...")
    tsp.animate_best_route(save_gif=False)
    
    return tsp, best_route, best_distance


if __name__ == "__main__":
    # Ejecutar ejemplo principal
    algorithm, route, distance = main()
    
    print(f"\n{'='*60}")
    print("RESULTADO FINAL:")
    print(f"Mejor ruta: {route}")
    print(f"Distancia total: {distance:.2f}")
    print(f"{'='*60}")
    print("ARCHIVOS REQUERIDOS (OPCIONALES):")
    print("- ciudades.csv: Coordenadas (x,y) de las ciudades")
    print("- distancias.csv: Matriz simétrica de distancias")
    print("Si no existen, el programa usa datos por defecto.")
    print(f"{'='*60}")