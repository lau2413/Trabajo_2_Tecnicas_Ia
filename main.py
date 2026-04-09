import sys
import os

# Asegurar que Python reconozca la carpeta src como un paquete
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importamos el trabajo de los 3 integrantes
# NOTA: Al importar data, se ejecutará su bloque de prueba e imprimirá el tamaño de un lote. Es normal.
from src.data import obtener_dataloaders
from src.modelo import entrenar_modelo
from src.optimizador import ejecutar_nas

def main():
    print("\n" + "="*60)
    print("🚀 INICIANDO INTEGRACIÓN DEL PROYECTO (NAS + CNN + DATOS)")
    print("="*60)
    
    # ---------------------------------------------------------
    # 1. OBTENER LOS DATOS (Persona 1)
    # ---------------------------------------------------------
    print("\n[1] Extrayendo los DataLoaders...")
    # Llamamos al generador de K-Fold de tu compañero
    generador_folds = obtener_dataloaders(batch_size=64, n_splits=5)
    
    # Extraemos solo el primer fold (train y val) para evaluar rápido los cromosomas
    train_loader, val_loader = next(generador_folds)
    print("✔️ DataLoaders extraídos correctamente.")

    # ---------------------------------------------------------
    # 2. CREAR LA FUNCIÓN PUENTE (Tu trabajo - Persona 2)
    # ---------------------------------------------------------
    print("\n[2] Configurando la función de Fitness con tu CNN...")
    
    def fitness_real(cromosoma):
        """
        Esta es la función que reemplaza el 'dummy_fitness' de tu compañero.
        Recibe un cromosoma, entrena tu modelo real con los datos reales,
        y devuelve el accuracy.
        """
        # Usamos 3 épocas por individuo para que la búsqueda inicial no tome horas
        return entrenar_modelo(
            cromosoma=cromosoma, 
            dataloader_train=train_loader, 
            dataloader_val=val_loader, 
            epocas=3 
        )

    # ---------------------------------------------------------
    # 3. EJECUTAR EL OPTIMIZADOR (Persona 3)
    # ---------------------------------------------------------
    print("\n[3] Iniciando la Búsqueda de Arquitectura Neuronal (NAS)...")
    
    # Pasamos tu función 'fitness_real' al código de tu compañero
    resultados = ejecutar_nas(
        tam_poblacion=4,      # 4 arquitecturas distintas por generación
        num_generaciones=2,   # 2 generaciones (para una prueba rápida)
        prob_mutacion=0.3,
        elitismo=1,
        fitness_fn=fitness_real, # <--- ¡Aquí ocurre la integración!
        verbose=True
    )

    # ---------------------------------------------------------
    # 4. RESULTADOS FINALES
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("🏆 BÚSQUEDA NAS FINALIZADA 🏆")
    print("="*60)
    print(f"Mejor Accuracy Encontrado: {resultados['mejor_fitness']:.4f}")
    print("Mejor Arquitectura Encontrada:")
    for key, value in resultados['mejor_cromosoma'].items():
        print(f"   - {key}: {value}")

if __name__ == "__main__":
    main()