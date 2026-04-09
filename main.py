import sys
import os

# Asegurar que Python reconozca la carpeta src como un paquete
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importamos el trabajo de los 3 integrantes
from src.data import obtener_dataloaders
from src.modelo import entrenar_modelo
from src.optimizador import ejecutar_nas

def main():
    print("="*60)
    print("🚀 INICIANDO INTEGRACIÓN DEL PROYECTO (NAS + CNN + DATOS)")
    print("="*60)
    
    # ---------------------------------------------------------
    # 1. CARGA DE DATOS (Persona 1)
    # ---------------------------------------------------------
    print("\n[1] Cargando datos y generando DataLoaders...")
    # Como NAS requiere entrenar muchísimos modelos, usar K-Fold completo para cada 
    # individuo sería computacionalmente inviable. Extraeremos solo el primer "fold" 
    # (un set de entrenamiento y uno de validación) para evaluar los cromosomas.
    generador_folds = obtener_dataloaders(batch_size=64, n_splits=5)
    train_loader, val_loader = next(generador_folds)
    print("✔️ DataLoaders listos.")

    # ---------------------------------------------------------
    # 2. FUNCIÓN PUENTE / FITNESS (Persona 2)
    # ---------------------------------------------------------
    print("\n[2] Configurando la función de Fitness...")
    
    def funcion_fitness(cromosoma):
        """
        Esta función es el puente: El optimizador le pasa un cromosoma, 
        aquí se entrena el modelo y se devuelve el accuracy para que 
        el optimizador sepa qué tan bueno es.
        """
        # Usamos 3 épocas para que la búsqueda no tarde días. 
        # Cuando quieran el modelo final definitivo, se pueden subir las épocas.
        accuracy = entrenar_modelo(
            cromosoma=cromosoma, 
            dataloader_train=train_loader, 
            dataloader_val=val_loader, 
            epocas=3 
        )
        return accuracy

    # ---------------------------------------------------------
    # 3. ALGORITMO EVOLUTIVO (Persona 3)
    # ---------------------------------------------------------
    print("\n[3] Iniciando la Búsqueda de Arquitectura Neuronal (NAS)...")
    
    # Valores pequeños para la primera prueba de integración
    resultados = ejecutar_nas(
        tam_poblacion=4,      # 4 individuos por generación
        num_generaciones=3,   # 3 generaciones
        prob_mutacion=0.3,
        elitismo=1,
        fitness_fn=funcion_fitness, # <--- Pasamos nuestra función puente
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