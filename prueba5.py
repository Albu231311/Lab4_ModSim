import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
from typing import List, Tuple, Union, Optional
import json
import csv
import os

class AlgoritmoGeneticoTSP:
    def __init__(self, 
                 tamPob: int = 100,
                 maxIter: int = 1000,
                 pctSobrev: float = 0.2,
                 pctCruce: float = 0.6,
                 pctMut: float = 0.2,
                 tasaMut: float = 0.1,
                 tamElite: int = 5):
        self.tamPob = tamPob
        self.maxIter = maxIter
        self.cantSobrev = int(tamPob * pctSobrev)
        self.cantCruce = int(tamPob * pctCruce)
        self.cantMut = int(tamPob * pctMut)
        self.tasaMut = tasaMut
        self.tamElite = min(tamElite, self.cantSobrev)
        
        total = self.cantSobrev + self.cantCruce + self.cantMut
        if total != tamPob:
            diff = tamPob - total
            self.cantCruce += diff
        
        self.ciudades = None
        self.matrizDist = None
        self.numCiudades = 0
        self.poblacion = []
        self.histFitness = []
        self.histMejorRuta = []
        self.histDiversidad = []
        self.mejorRutaGlobal: Optional[List[int]] = None
        self.mejorDistGlobal: float = float('inf')
        
    def cargarCoordsArchivo(self, archivo: str):
        try:
            if archivo.endswith('.json'):
                with open(archivo, 'r') as f:
                    data = json.load(f)
                    self.ciudades = np.array(data)
            elif archivo.endswith('.csv'):
                self.ciudades = np.loadtxt(archivo, delimiter=',', comments='#')
            else:
                self.ciudades = np.loadtxt(archivo, delimiter=',', comments='#')
                
            self.numCiudades = len(self.ciudades)
            self._calcMatrizDist()
            return True
        except Exception as e:
            return False
            
    def cargarCoordsLista(self, coords: List[Tuple[float, float]]):
        self.ciudades = np.array(coords)
        self.numCiudades = len(self.ciudades)
        self._calcMatrizDist()
        
    def cargarMatrizDistArchivo(self, archivo: str):
        try:
            if archivo.endswith('.json'):
                with open(archivo, 'r') as f:
                    self.matrizDist = np.array(json.load(f))
            else:
                self.matrizDist = np.loadtxt(archivo, delimiter=',', comments='#')
                
            self.matrizDist = np.array(self.matrizDist, dtype=float)
            if self.matrizDist.shape[0] != self.matrizDist.shape[1]:
                raise ValueError("La matriz debe ser cuadrada")
            if not np.allclose(self.matrizDist, self.matrizDist.T, atol=1e-6):
                self.matrizDist = (self.matrizDist + self.matrizDist.T) / 2.0
            np.fill_diagonal(self.matrizDist, 0.0)

            self.numCiudades = len(self.matrizDist)
            self._genCoordsAprox()
            return True
        except Exception as e:
            return False
            
    def cargarMatrizDistArray(self, matriz: np.ndarray):
        self.matrizDist = np.array(matriz, dtype=float)
        if self.matrizDist.shape[0] != self.matrizDist.shape[1]:
            raise ValueError("La matriz debe ser cuadrada")
        if not np.allclose(self.matrizDist, self.matrizDist.T, atol=1e-6):
            self.matrizDist = (self.matrizDist + self.matrizDist.T) / 2.0
        np.fill_diagonal(self.matrizDist, 0.0)

        self.numCiudades = len(self.matrizDist)
        self._genCoordsAprox()
        
    def _calcMatrizDist(self):
        self.matrizDist = np.zeros((self.numCiudades, self.numCiudades))
        for i in range(self.numCiudades):
            for j in range(self.numCiudades):
                if i != j:
                    dist = np.sqrt((self.ciudades[i][0] - self.ciudades[j][0])**2 + 
                                 (self.ciudades[i][1] - self.ciudades[j][1])**2)
                    self.matrizDist[i][j] = dist
                    
    def _genCoordsAprox(self):
        if self.matrizDist is None:
            angles = np.linspace(0, 2*np.pi, self.numCiudades, endpoint=False)
            radio = 10
            self.ciudades = np.column_stack([radio * np.cos(angles), radio * np.sin(angles)])
            return

        D = np.array(self.matrizDist, dtype=float)
        n = self.numCiudades
        D2 = D ** 2

        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J.dot(D2).dot(J)

        evals, evecs = np.linalg.eigh(B)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        if np.any(evals[:2] < -1e-6):
            pass

        L = np.diag(np.sqrt(np.clip(evals[:2], a_min=0, a_max=None)))
        coords = evecs[:, :2].dot(L)

        coords -= coords.mean(axis=0)
        self.ciudades = coords
        
    def _inicPoblacion(self):
        self.poblacion = []
        
        for _ in range(self.tamPob // 2):
            ruta = list(range(self.numCiudades))
            random.shuffle(ruta)
            self.poblacion.append(ruta)
            
        for _ in range(self.tamPob - len(self.poblacion)):
            ruta = self._heuristicaVecino(random.randint(0, self.numCiudades-1))
            for _ in range(random.randint(1, 5)):
                ruta = self._mutar2opt(ruta.copy())
            self.poblacion.append(ruta)
            
    def _heuristicaVecino(self, ciudadInicio: int) -> List[int]:
        ruta = [ciudadInicio]
        restantes = set(range(self.numCiudades)) - {ciudadInicio}
        
        actual = ciudadInicio
        while restantes:
            cercana = min(restantes, key=lambda x: self.matrizDist[actual][x])
            ruta.append(cercana)
            restantes.remove(cercana)
            actual = cercana
            
        return ruta
        
    def _calcFitness(self, ruta: List[int]) -> float:
        distTotal = 0
        for i in range(len(ruta)):
            deCiudad = ruta[i]
            aCiudad = ruta[(i + 1) % len(ruta)]
            distTotal += self.matrizDist[deCiudad][aCiudad]
        return 1 / (distTotal + 1e-8)
        
    def _calcDistancia(self, ruta: List[int]) -> float:
        distTotal = 0
        for i in range(len(ruta)):
            deCiudad = ruta[i]
            aCiudad = ruta[(i + 1) % len(ruta)]
            distTotal += self.matrizDist[deCiudad][aCiudad]
        return distTotal
        
    def _seleccionTorneo(self, poblacion: List[List[int]], k: int = 5) -> List[int]:
        torneo = random.sample(poblacion, min(k, len(poblacion)))
        return max(torneo, key=self._calcFitness)
        
    def _seleccionRuleta(self, poblacion: List[List[int]]) -> List[int]:
        scores = [self._calcFitness(ruta) for ruta in poblacion]
        totalFit = sum(scores)
        
        if totalFit == 0:
            return random.choice(poblacion)
            
        probs = [f/totalFit for f in scores]
        
        r = random.random()
        probAcum = 0.0
        for i, prob in enumerate(probs):
            probAcum += prob
            if r <= probAcum:
                return poblacion[i]
        
        return poblacion[-1]
        
    def _cruceOx(self, padre1: List[int], padre2: List[int]) -> Tuple[List[int], List[int]]:
        tam = len(padre1)
        inicio, fin = sorted(random.sample(range(tam), 2))
        
        hijo1 = [-1] * tam
        hijo2 = [-1] * tam
        
        hijo1[inicio:fin] = padre1[inicio:fin]
        hijo2[inicio:fin] = padre2[inicio:fin]
        
        self._llenarHijo(hijo1, padre2, inicio, fin)
        self._llenarHijo(hijo2, padre1, inicio, fin)
        
        return hijo1, hijo2
        
    def _llenarHijo(self, hijo: List[int], padre: List[int], inicio: int, fin: int):
        hijoSet = set(hijo[inicio:fin])
        padreFilt = [gen for gen in padre if gen not in hijoSet]
        
        idx = 0
        for i in range(inicio):
            hijo[i] = padreFilt[idx]
            idx += 1
            
        for i in range(fin, len(hijo)):
            hijo[i] = padreFilt[idx]
            idx += 1
            
    def _crucePmx(self, padre1: List[int], padre2: List[int]) -> Tuple[List[int], List[int]]:
        tam = len(padre1)
        inicio, fin = sorted(random.sample(range(tam), 2))
        
        hijo1 = padre1.copy()
        hijo2 = padre2.copy()
        
        hijo1[inicio:fin], hijo2[inicio:fin] = padre2[inicio:fin], padre1[inicio:fin]
        
        self._resolverConflictosPmx(hijo1, padre1, padre2, inicio, fin)
        self._resolverConflictosPmx(hijo2, padre2, padre1, inicio, fin)
        
        return hijo1, hijo2
        
    def _resolverConflictosPmx(self, hijo: List[int], padre1: List[int], padre2: List[int], 
                              inicio: int, fin: int):
        for i in range(len(hijo)):
            if i < inicio or i >= fin:
                while hijo[i] in hijo[inicio:fin]:
                    pos = padre2.index(hijo[i])
                    hijo[i] = padre1[pos]
                    
    def _mutar2opt(self, ruta: List[int]) -> List[int]:
        if len(ruta) < 4:
            return ruta
            
        ruta = ruta.copy()
        i, j = sorted(random.sample(range(len(ruta)), 2))
        if j - i < 2:
            return ruta
            
        ruta[i:j+1] = ruta[i:j+1][::-1]
        return ruta
        
    def _mutarSwap(self, ruta: List[int]) -> List[int]:
        ruta = ruta.copy()
        i, j = random.sample(range(len(ruta)), 2)
        ruta[i], ruta[j] = ruta[j], ruta[i]
        return ruta
        
    def _mutarScramble(self, ruta: List[int]) -> List[int]:
        ruta = ruta.copy()
        inicio = random.randint(0, len(ruta) - 3)
        fin = random.randint(inicio + 2, len(ruta))
        segmento = ruta[inicio:fin]
        random.shuffle(segmento)
        ruta[inicio:fin] = segmento
        return ruta
        
    def _calcDiversidad(self) -> float:
        if len(self.poblacion) < 2:
            return 0.0
            
        totalDist = 0
        comparaciones = 0
        
        for i in range(min(20, len(self.poblacion))):
            for j in range(i + 1, min(20, len(self.poblacion))):
                diff = sum(1 for a, b in zip(self.poblacion[i], self.poblacion[j]) if a != b)
                totalDist += diff
                comparaciones += 1
                
        return totalDist / comparaciones if comparaciones > 0 else 0.0
        
    def _mantenerDiversidad(self):
        diversidad = self._calcDiversidad()
        
        if diversidad < 0.1:
            numReemplazo = self.tamPob // 4
            scores = [(i, self._calcFitness(ruta)) 
                            for i, ruta in enumerate(self.poblacion)]
            scores.sort(key=lambda x: x[1])
            
            for i in range(numReemplazo):
                idx = scores[i][0]
                self.poblacion[idx] = self._heuristicaVecino(random.randint(0, self.numCiudades-1))
                
    def _busquedaLocal2opt(self, ruta: List[int]) -> List[int]:
        if ruta is None:
            return ruta
        mejor = ruta.copy()
        mejora = True
        mejorDist = self._calcDistancia(mejor)

        while mejora:
            mejora = False
            n = len(mejor)
            for i in range(0, n - 1):
                for j in range(i + 1, n):
                    if j - i == 1: continue
                    nueva = mejor.copy()
                    nueva[i:j] = nueva[i:j][::-1]
                    nuevaDist = self._calcDistancia(nueva)
                    if nuevaDist < mejorDist:
                        mejor = nueva
                        mejorDist = nuevaDist
                        mejora = True
        return mejor
                
    def resolver(self, verbose: bool = True) -> Tuple[List[int], float]:
        if self.matrizDist is None:
            raise ValueError("No se ha cargado información de ciudades o distancias")

        if self.numCiudades < 2:
            raise ValueError("Se requieren al menos 2 ciudades para resolver TSP")

        self.mejorRutaGlobal = None
        self.mejorDistGlobal = float('inf')

        self._inicPoblacion()
        self.histFitness = []
        self.histMejorRuta = []
        self.histDiversidad = []
        
        for generacion in range(self.maxIter):
            scores = [(ruta, self._calcFitness(ruta)) 
                            for ruta in self.poblacion]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            mejorRuta = scores[0][0]
            mejorDist = self._calcDistancia(mejorRuta)
            self.histFitness.append(mejorDist)
            self.histMejorRuta.append(mejorRuta.copy())

            if mejorDist < self.mejorDistGlobal:
                self.mejorDistGlobal = mejorDist
                self.mejorRutaGlobal = mejorRuta.copy()
            
            diversidad = self._calcDiversidad()
            self.histDiversidad.append(diversidad)
            
            if verbose and generacion % 50 == 0:
                print(f"Gen {generacion}: Mejor dist = {mejorDist:.2f}")
                      
            nuevaPob = []
            
            for i in range(self.tamElite):
                nuevaPob.append(scores[i][0].copy())
                
            for _ in range(self.cantSobrev - self.tamElite):
                seleccionado = self._seleccionTorneo(self.poblacion)
                nuevaPob.append(seleccionado.copy())
                
            for _ in range(self.cantCruce // 2):
                p1 = self._seleccionRuleta(self.poblacion)
                p2 = self._seleccionTorneo(self.poblacion)
                h1, h2 = self._cruceOx(p1, p2)
                nuevaPob.extend([h1, h2])
                    
            for _ in range(self.cantMut):
                individuo = self._seleccionRuleta(self.poblacion)
                if random.random() < self.tasaMut:
                    mutado = self._mutar2opt(individuo)
                else:
                    mutado = self._mutarSwap(individuo)
                nuevaPob.append(mutado)
                
            if not nuevaPob:
                nuevaPob = [list(range(self.numCiudades))]
            while len(nuevaPob) < self.tamPob:
                nuevaPob.append(self._heuristicaVecino(random.randint(0, self.numCiudades-1)))
            if len(nuevaPob) > self.tamPob:
                nuevaPob = nuevaPob[:self.tamPob]
                
            self.poblacion = nuevaPob
            self._mantenerDiversidad()
                
        mejorRuta = self.mejorRutaGlobal
        mejorDist = self.mejorDistGlobal

        if mejorRuta is not None:
            mejorRuta = self._busquedaLocal2opt(mejorRuta)
            mejorDist = self._calcDistancia(mejorRuta)

        if verbose:
            print(f"\nMejor ruta encontrada: {mejorRuta}")
            print(f"Distancia total: {mejorDist:.2f}")
            
        return mejorRuta, mejorDist
        
    def animarMejorRuta(self, guardarGif: bool = False, nombreArchivo: str = "tsp_mejor_ruta.gif"):
        if not self.histMejorRuta and self.mejorRutaGlobal is None:
            print("No hay datos de rutas para animar")
            return None
            
        if self.mejorRutaGlobal is not None:
            mejorRuta = self.mejorRutaGlobal
        else:
            mejorRuta = self.histMejorRuta[-1]
            
        fig, ax = plt.subplots(figsize=(12, 9))
        
        def animar(frame):
            ax.clear()
            
            if frame < len(mejorRuta):
                rutaParcial = mejorRuta[:frame+1]
                
                if len(rutaParcial) > 1:
                    x = [self.ciudades[ciudad][0] for ciudad in rutaParcial]
                    y = [self.ciudades[ciudad][1] for ciudad in rutaParcial]
                    ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
                
                x = [self.ciudades[ciudad][0] for ciudad in rutaParcial]
                y = [self.ciudades[ciudad][1] for ciudad in rutaParcial]
                ax.scatter(x, y, c='red', s=100, zorder=5)
                
                for i, ciudad in enumerate(rutaParcial):
                    ax.annotate(str(ciudad), (self.ciudades[ciudad][0], self.ciudades[ciudad][1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=10, color='black')
            else:
                x = [self.ciudades[ciudad][0] for ciudad in mejorRuta]
                y = [self.ciudades[ciudad][1] for ciudad in mejorRuta]
                x.append(x[0])
                y.append(y[0])
                ax.plot(x, y, 'b-', linewidth=2)
                
                ax.scatter([self.ciudades[ciudad][0] for ciudad in mejorRuta], 
                          [self.ciudades[ciudad][1] for ciudad in mejorRuta], 
                          c='red', s=100, zorder=5)
                
                for ciudad in mejorRuta:
                    ax.annotate(str(ciudad), (self.ciudades[ciudad][0], self.ciudades[ciudad][1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=10, color='black')
            
            ax.set_title(f'TSP - Construcción de la Mejor Ruta\nDistancia: {self.mejorDistGlobal:.2f}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
        
        totalFrames = len(mejorRuta) + 20
        ani = animation.FuncAnimation(fig, animar, frames=totalFrames, 
                                    interval=800, repeat=True, blit=False)
        
        plt.subplots_adjust(top=0.9)
        
        if guardarGif:
            ani.save(nombreArchivo, writer='pillow', fps=1.5)

        plt.show()
        return ani


def obtenerEleccionArchivo():
    print("\n--- Selección de archivos ---")
    print("1. Archivo de coordenadas (ciudades.csv)")
    print("2. Archivo de matriz de distancias (distancias.csv)")
    
    try:
        eleccion = input("Seleccione opción (1-2) o presione enter para usar archivos por defecto: ").strip()

        if eleccion == '1' or eleccion == '':
            return "ciudades.csv", None
        elif eleccion == '2':
            return None, "distancias.csv"
        else:
            print("Opción inválida. Se usarán archivos por defecto")
            return "ciudades.csv", None
            
    except Exception:
        return "ciudades.csv", None


def obtenerParametrosUsuario():
    print("\n--- Configuración del algoritmo ---")
    print("Presiona ENTER para usar valores por defecto")
    
    try:
        tamPob = input("Tamaño de población (defecto: 50): ").strip()
        tamPoblacion = int(tamPob) if tamPob else 50

        maxIter = input("Número máximo de iteraciones (defecto: 200): ").strip()
        maxIteraciones = int(maxIter) if maxIter else 200

        pctSobrev = input("Porcentaje de sobrevivientes (defecto: 0.3): ").strip()
        pctSobrevivientes = float(pctSobrev) if pctSobrev else 0.3

        pctCruce = input("Porcentaje de población por cruce (defecto: 0.5): ").strip()
        pctCruces = float(pctCruce) if pctCruce else 0.5

        pctMut = input("Porcentaje de población por mutación (defecto: 0.2): ").strip()
        pctMutaciones = float(pctMut) if pctMut else 0.2

        tasaMut = input("Tasa de mutación (defecto: 0.15): ").strip()
        tasaMutacion = float(tasaMut) if tasaMut else 0.15
        
        if tamPoblacion < 10:
            tamPoblacion = 10
            
        if pctSobrevivientes + pctCruces + pctMutaciones > 1.5:
            pctSobrevivientes = 0.3
            pctCruces = 0.5
            pctMutaciones = 0.2
        
    except ValueError:
        tamPoblacion = 50
        maxIteraciones = 200
        pctSobrevivientes = 0.3
        pctCruces = 0.5
        pctMutaciones = 0.2
        tasaMutacion = 0.15
    
    return {
        'tamPob': tamPoblacion,
        'maxIter': maxIteraciones,
        'pctSobrev': pctSobrevivientes,
        'pctCruce': pctCruces,
        'pctMut': pctMutaciones,
        'tasaMut': tasaMutacion,
        'tamElite': max(3, int(tamPoblacion * 0.05))
    }


def main():
    print("--- Algoritmo Genético para TSP ---")

    archivoCoords, archivoDist = obtenerEleccionArchivo()
    params = obtenerParametrosUsuario()
    
    tsp = AlgoritmoGeneticoTSP(**params)
    
    datosCargados = False
    
    if archivoCoords:
        try:
            datosCargados = tsp.cargarCoordsArchivo(archivoCoords)
        except Exception as e:
            pass
    
    if not datosCargados and archivoDist:
        try:
            datosCargados = tsp.cargarMatrizDistArchivo(archivoDist)
        except Exception as e:
            pass
    
    if not datosCargados:
        ciudadesDefault = [
            (12.5, 87.3), (45.2, 23.8), (78.9, 65.4), (23.1, 45.7), (89.6, 12.3),
            (34.7, 78.9), (67.8, 34.5), (15.4, 56.2), (92.3, 89.7), (56.7, 8.9),
            (8.2, 34.6), (73.4, 91.2), (28.9, 67.8), (85.1, 45.3), (41.6, 72.1)
        ]
        tsp.cargarCoordsLista(ciudadesDefault)
    
    mejorRuta, mejorDist = tsp.resolver(verbose=True)
    
    print("\nSim...")
    tsp.animarMejorRuta(guardarGif=False)
    
    return tsp, mejorRuta, mejorDist


if __name__ == "__main__":
    try:
        algoritmo, ruta, distancia = main()
        
        print(f"\n{'='*60}")
        print("Resultado final:")
        print(f"Mejor ruta: {ruta}")
        print(f"Distancia total: {distancia:.2f}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error en la ejecución: {e}")
