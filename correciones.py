# correciones.py
# Juego de la vida de Conway optimizado para medicion de rendimiento
# Enfoque especial en escalamiento debil para mantener tiempos constantes con diferentes procesos

import numpy as np
import multiprocessing as mp
import time
import os
import matplotlib.pyplot as plt
import cProfile
import pstats

# ---------------------------------------------------
# Clase que implementa el Juego de la Vida
# ---------------------------------------------------
class JuegoDeLaVida:
    def __init__(self, filas, columnas, pasos, procesos):
        # Guardamos parametros basicos
        self.filas = filas
        self.columnas = columnas
        self.pasos = pasos
        self.procesos = procesos
        
        # Creamos grilla inicial aleatoria de 0s y 1s
        self.grilla = np.random.randint(2, size=(filas, columnas), dtype=np.uint8)
        
    def contar_vecinos(self, subgrilla):
        # Calcula la cantidad de vecinos vivos usando convoluciones via desplazamientos
        vecinos = (
            np.roll(np.roll(subgrilla, 1, 0), 1, 1) +   # arriba izquierda
            np.roll(np.roll(subgrilla, 1, 0), 0, 1) +   # arriba
            np.roll(np.roll(subgrilla, 1, 0), -1, 1) +  # arriba derecha
            np.roll(np.roll(subgrilla, 0, 0), 1, 1) +   # izquierda
            np.roll(np.roll(subgrilla, 0, 0), -1, 1) +  # derecha
            np.roll(np.roll(subgrilla, -1, 0), 1, 1) +  # abajo izquierda
            np.roll(np.roll(subgrilla, -1, 0), 0, 1) +  # abajo
            np.roll(np.roll(subgrilla, -1, 0), -1, 1)   # abajo derecha
        )
        return vecinos

    def step(self):
        # Calcula una generacion completa
        vecinos = self.contar_vecinos(self.grilla)
        # Aplica las reglas del juego
        self.grilla = ((vecinos == 3) | ((self.grilla == 1) & (vecinos == 2))).astype(np.uint8)

    def run(self):
        # Ejecuta todos los pasos del juego
        for _ in range(self.pasos):
            self.step()

# ---------------------------------------------------
# Funcion auxiliar para ejecucion paralela
# ---------------------------------------------------
def worker(subgrilla, pasos):
    juego = JuegoDeLaVida(*subgrilla.shape, pasos, 1)
    juego.grilla = subgrilla
    for _ in range(pasos):
        vecinos = juego.contar_vecinos(juego.grilla)
        juego.grilla = ((vecinos == 3) | ((juego.grilla == 1) & (vecinos == 2))).astype(np.uint8)
    return juego.grilla

# ---------------------------------------------------
# Escalamiento debil
# ---------------------------------------------------
def escalamiento_debil(celdas_por_proceso, pasos, lista_procesos):
    tiempos = []
    for p in lista_procesos:
        # Tamaño proporcional al numero de procesos
        lado = int(np.sqrt(celdas_por_proceso * p))
        juego = JuegoDeLaVida(lado, lado, pasos, p)
        
        # Dividimos grilla en partes casi iguales
        subgrillas = np.array_split(juego.grilla, p, axis=0)
        
        # Calentamiento previo
        _ = worker(subgrillas[0], 1)

        # Ejecucion cronometrada
        inicio = time.time()
        with mp.Pool(processes=p) as pool:
            pool.starmap(worker, [(sg, pasos) for sg in subgrillas])
        fin = time.time()
        
        tiempo = fin - inicio
        tiempos.append(tiempo)
        print(f">> {p} procesos con grilla {lado}x{lado}")
        print(f"Tiempo: {tiempo:.4f} s")
    return tiempos

# ---------------------------------------------------
# Graficar escalamiento debil
# ---------------------------------------------------
def graficar_escalamiento_debil(procesos, tiempos):
    plt.figure()
    plt.plot(procesos, tiempos, marker='o')
    plt.xlabel('Numero de procesos')
    plt.ylabel('Tiempo de ejecucion (s)')
    plt.title('Escalamiento debil')
    plt.grid(True)
    os.makedirs("graficas", exist_ok=True)
    plt.savefig("graficas/escalamiento_debil.png")
    plt.close()

# ---------------------------------------------------
# Main para perfilado y ejecucion
# ---------------------------------------------------
if __name__ == "__main__":
    procesos = [1, 2, 4, 8]
    pasos = 50
    celdas_por_proceso = 250000  # carga constante por proceso
    pr = cProfile.Profile()
    pr.enable()
    
    tiempos = escalamiento_debil(celdas_por_proceso, pasos, procesos)
    
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.dump_stats("perfilado_debil.pstats")
    
    graficar_escalamiento_debil(procesos, tiempos)
    
    # Guardar resultados en archivo para informe
    with open("resultados_escalamiento_debil.txt", "w") as f:
        for p, t in zip(procesos, tiempos):
            f.write(f"{p} procesos: {t:.4f} s\n")