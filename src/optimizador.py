# -*- coding: utf-8 -*-

#formato exacto que espera CNN_Dinamica de la Persona 2:
#   {
#       'conv_filters': list[int],
#       'kernel_sizes': list[int],
#       'dense_units':  list[int],
#       'lr':           float,
#   }

import random
import copy
import math

# 1. ESPACIO DE BÚSQUEDA

N_CONV_OPCIONES   = [1, 2, 3]           # Cuántos bloques convolucionales
FILTROS_OPCIONES  = [16, 32, 64, 128]   # Filtros por capa conv
KERNEL_OPCIONES   = [3, 5]              # Tamaño de kernel por capa conv
DENSE_N_OPCIONES  = [1, 2]             # Cuántas capas densas
DENSE_U_OPCIONES  = [64, 128, 256, 512] # Unidades por capa densa
LR_OPCIONES       = [1e-4, 5e-4, 1e-3, 5e-3]


# 2. FUNCIÓN SUSTITUTA (dummy_fitness)

def dummy_fitness(cromosoma: dict, **kwargs) -> float:
    """
    Función sustituta de fitness.
    Retorna un accuracy simulado (aleatorio con pequeño sesgo).

    Para la integración final — se debe reemplazar la llamada en ejecutar_nas():
        from src.modelo import entrenar_modelo
        resultado = ejecutar_nas(
            fitness_fn=lambda c: entrenar_modelo(c, train_loader, val_loader)
        )
    """
    base = random.uniform(0.40, 0.92)
    bonus = 0.0

    bonus += len(cromosoma.get("conv_filters", [])) * 0.01
    bonus += len(cromosoma.get("dense_units",  [])) * 0.005
    # Learning rate moderado
    if cromosoma.get("lr") in [5e-4, 1e-3]:
        bonus += 0.01
    return round(min(base + bonus, 1.0), 6)

# 3. INICIALIZACIÓN DE POBLACIÓN

def crear_cromosoma() -> dict:
    """
    Genera un cromosoma aleatorio compatible con CNN_Dinamica (creado por la persona 2).
    conv_filters y kernel_sizes tienen siempre la misma longitud.
    """
    n_conv  = random.choice(N_CONV_OPCIONES)
    n_dense = random.choice(DENSE_N_OPCIONES)

    cromosoma = {
        "conv_filters": [random.choice(FILTROS_OPCIONES) for _ in range(n_conv)],
        "kernel_sizes": [random.choice(KERNEL_OPCIONES)  for _ in range(n_conv)],
        "dense_units":  [random.choice(DENSE_U_OPCIONES) for _ in range(n_dense)],
        "lr":           random.choice(LR_OPCIONES),
    }
    return cromosoma


def inicializar_poblacion(tam_poblacion: int = 10) -> list:
    """
    Genera la población inicial.

    Args:
        tam_poblacion: Número de individuos.

    Returns:
        Lista de cromosomas.
    """
    poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
    print(f"[NAS] Poblacion inicial creada: {tam_poblacion} individuos.")
    return poblacion

# 4. OPERADORES GENÉTICOS

def crossover(padre1: dict, padre2: dict) -> tuple:
    """
    Cruce uniforme por gen:
    Cada gen del hijo se toma aleatoriamente de uno de los dos padres.
    Es más robusto que el cruce de un punto cuando los genes son listas
    de longitud variable (como está conv_filters).

    Args:
        padre1, padre2: Cromosomas padres.

    Returns:
        Tupla (hijo1, hijo2).
    """
    genes = list(padre1.keys())

    hijo1 = {}
    hijo2 = {}
    for gen in genes:
        if random.random() < 0.5:
            hijo1[gen] = copy.deepcopy(padre1[gen])
            hijo2[gen] = copy.deepcopy(padre2[gen])
        else:
            hijo1[gen] = copy.deepcopy(padre2[gen])
            hijo2[gen] = copy.deepcopy(padre1[gen])

    # Garantizar coherencia: conv_filters y kernel_sizes deben tener igual longitud
    hijo1 = _reparar_cromosoma(hijo1)
    hijo2 = _reparar_cromosoma(hijo2)

    return hijo1, hijo2


def _reparar_cromosoma(cromosoma: dict) -> dict:
    """
    Tras un cruce, conv_filters y kernel_sizes pueden quedar con distinta
    longitud. Esta función iguala kernel_sizes a conv_filters truncando o
    rellenando con valores aleatorios.
    """
    n_filtros = len(cromosoma["conv_filters"])
    n_kernels = len(cromosoma["kernel_sizes"])

    if n_filtros > n_kernels:
        for _ in range(n_filtros - n_kernels):
            cromosoma["kernel_sizes"].append(random.choice(KERNEL_OPCIONES))
    elif n_kernels > n_filtros:
        cromosoma["kernel_sizes"] = cromosoma["kernel_sizes"][:n_filtros]

    # Si quedó sin capas convolucionales, agregar al menos una
    if n_filtros == 0:
        cromosoma["conv_filters"] = [random.choice(FILTROS_OPCIONES)]
        cromosoma["kernel_sizes"] = [random.choice(KERNEL_OPCIONES)]

    return cromosoma


def mutar(cromosoma: dict, prob_mutacion: float = 0.3) -> dict:
    """
    Mutación gen a gen:
    * 'conv_filters' / 'kernel_sizes': con prob_mutacion, cambia uno de sus
      elementos al azar, agrega una capa, o elimina la última (si hay > 1).
    * 'dense_units': con prob_mutacion, cambia uno de sus elementos.
    * 'lr': con prob_mutacion, elige un nuevo valor del espacio de búsqueda.

    Args:
        cromosoma: Cromosoma original (no se modifica).
        prob_mutacion: Probabilidad de mutación por gen.

    Returns:
        Nuevo cromosoma mutado.
    """
    m = copy.deepcopy(cromosoma)

    # Mutar conv_filters (y sincronizar kernel_sizes)
    if random.random() < prob_mutacion:
        accion = random.choice(["cambiar", "agregar", "eliminar"])
        if accion == "cambiar" and m["conv_filters"]:
            idx = random.randrange(len(m["conv_filters"]))
            m["conv_filters"][idx] = random.choice(FILTROS_OPCIONES)
            m["kernel_sizes"][idx] = random.choice(KERNEL_OPCIONES)
        elif accion == "agregar" and len(m["conv_filters"]) < max(N_CONV_OPCIONES):
            m["conv_filters"].append(random.choice(FILTROS_OPCIONES))
            m["kernel_sizes"].append(random.choice(KERNEL_OPCIONES))
        elif accion == "eliminar" and len(m["conv_filters"]) > 1:
            m["conv_filters"].pop()
            m["kernel_sizes"].pop()

    # Mutar dense_units
    if random.random() < prob_mutacion and m["dense_units"]:
        idx = random.randrange(len(m["dense_units"]))
        m["dense_units"][idx] = random.choice(DENSE_U_OPCIONES)

    # Mutar learning rate
    if random.random() < prob_mutacion:
        m["lr"] = random.choice(LR_OPCIONES)

    return m

# 5. SELECCIÓN POR TORNEO

def seleccion_torneo(poblacion: list, fitnesses: list, k: int = 3) -> dict:
    """
    Selección por torneo: escoge k individuos al azar y retorna el mejor.

    Args:
        poblacion:  Lista de cromosomas.
        fitnesses:  Valores de fitness correspondientes.
        k:          Tamaño del torneo.

    Returns:
        Copia del cromosoma ganador.
    """
    participantes = random.sample(range(len(poblacion)), min(k, len(poblacion)))
    ganador = max(participantes, key=lambda i: fitnesses[i])
    return copy.deepcopy(poblacion[ganador])

# 6. BUCLE EVOLUTIVO PRINCIPAL

def ejecutar_nas(
    tam_poblacion:    int   = 10,
    num_generaciones: int   = 5,
    prob_mutacion:    float = 0.3,
    elitismo:         int   = 2,
    fitness_fn              = None,
    verbose:          bool  = True,
) -> dict:
    """
    Ciclo principal del algoritmo evolutivo NAS.

    Args:
        tam_poblacion:    Individuos por generación.
        num_generaciones: Generaciones a evolucionar.
        prob_mutacion:    Probabilidad de mutación por gen.
        elitismo:         Mejores individuos que pasan sin cambios.
        fitness_fn:       Función de evaluación.
                          None  → usa dummy_fitness (modo prueba).
                          Para integración final:
                              from src.modelo import entrenar_modelo
                              fitness_fn = lambda c: entrenar_modelo(
                                  c, train_loader, val_loader
                              )
        verbose:          Imprime el progreso.

    Returns:
        Dict con 'mejor_cromosoma', 'mejor_fitness' e 'historial'.
    """
    if fitness_fn is None:
        fitness_fn = dummy_fitness

    poblacion        = inicializar_poblacion(tam_poblacion)
    mejor_global     = None
    mejor_fit_global = -math.inf
    historial        = []

    for gen_num in range(1, num_generaciones + 1):
        if verbose:
            print(f"\n[NAS] -- Generacion {gen_num}/{num_generaciones} --")

        # -- Evaluación ------------------------------------------------------
        fitnesses = []
        for idx, cromosoma in enumerate(poblacion):
            f = fitness_fn(cromosoma)
            fitnesses.append(f)
            if verbose:
                print(
                    f"  [{idx+1:02d}] acc={f:.4f} | "
                    f"conv_filters={cromosoma['conv_filters']} | "
                    f"kernel_sizes={cromosoma['kernel_sizes']} | "
                    f"dense_units={cromosoma['dense_units']} | "
                    f"lr={cromosoma['lr']}"
                )

        # -- Mejor de la generación ------------------------------------------
        mejor_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        mejor_fit = fitnesses[mejor_idx]
        promedio  = sum(fitnesses) / len(fitnesses)

        historial.append({
            "generacion":       gen_num,
            "mejor_fitness":    mejor_fit,
            "promedio_fitness": promedio,
        })

        if mejor_fit > mejor_fit_global:
            mejor_fit_global = mejor_fit
            mejor_global     = copy.deepcopy(poblacion[mejor_idx])

        if verbose:
            print(f"  >> Mejor gen {gen_num}: {mejor_fit:.4f} | "
                  f"Promedio: {promedio:.4f} | "
                  f"Mejor global: {mejor_fit_global:.4f}")

        # -- Elitismo --------------------------------------------------------
        orden           = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        nueva_poblacion = [copy.deepcopy(poblacion[i]) for i in orden[:elitismo]]

        # -- Reproducción ----------------------------------------------------
        while len(nueva_poblacion) < tam_poblacion:
            padre1       = seleccion_torneo(poblacion, fitnesses)
            padre2       = seleccion_torneo(poblacion, fitnesses)
            hijo1, hijo2 = crossover(padre1, padre2)
            hijo1        = mutar(hijo1, prob_mutacion)
            hijo2        = mutar(hijo2, prob_mutacion)
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < tam_poblacion:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

    # -- Resultado final ------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("[NAS] Busqueda completada.")
        print(f"[NAS] Mejor fitness global: {mejor_fit_global:.4f}")
        print("[NAS] Mejor cromosoma encontrado:")
        for k, v in mejor_global.items():
            print(f"       {k}: {v}")
        print("=" * 60)

    return {
        "mejor_cromosoma": mejor_global,
        "mejor_fitness":   mejor_fit_global,
        "historial":       historial,
    }

# 7. PUNTO DE ENTRADA — prueba standalone con dummy_fitness

if __name__ == "__main__":
    print("=" * 60)
    print("  NAS Optimizer — Detección de Lenguaje de Señas")
    print("  Persona 3 | Modo Integración")
    print("=" * 60)

    from data import obtener_dataloaders
    from modelo import entrenar_modelo

    #Se toma el primer fold para evaluar cada cromosoma
    train_loader, val_loader = next(iter(obtener_dataloaders(batch_size=64, n_splits=5)))

    # ── Ejecutar NAS con modelo real ───────────────────────────
    resultado = ejecutar_nas(
        tam_poblacion=4,
        num_generaciones=3,
        prob_mutacion=0.3,
        elitismo=2,
        fitness_fn=lambda c: entrenar_modelo(c, train_loader, val_loader),
        verbose=True,
    )

    print("\n[NAS] Historial de evolución:")
    print(f"  {'Gen':>4} | {'Mejor':>8} | {'Promedio':>8}")
    print("  " + "-" * 28)
    for h in resultado["historial"]:
        print(f"  {h['generacion']:>4} | {h['mejor_fitness']:>8.4f} | {h['promedio_fitness']:>8.4f}")