import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. ARQUITECTURA DINÁMICA DE LA CNN
# ==========================================
class CNN_Dinamica(nn.Module):
    def __init__(self, cromosoma, num_clases=26):
        """
        cromosoma: Diccionario con la configuración de la red.
        Ejemplo: {'conv_filters': [32, 64], 'kernel_sizes': [3, 3], 'dense_units': [128]}
        """
        super(CNN_Dinamica, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = 1 # Imágenes en escala de grises (1 canal)
        
        # Construir bloques convolucionales dinámicamente
        filtros = cromosoma.get('conv_filters', [])
        kernels = cromosoma.get('kernel_sizes', [])
        
        for out_channels, k_size in zip(filtros, kernels):

            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding='same'))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels # La salida de esta capa es la entrada de la siguiente
            
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Construir capas densas dinámicamente
        self.dense_layers = nn.ModuleList()
        in_features = in_channels # Los canales restantes tras el Global Pooling
        unidades_densas = cromosoma.get('dense_units', [])
        
        for units in unidades_densas:
            self.dense_layers.append(nn.Linear(in_features, units))
            self.dense_layers.append(nn.ReLU())
            in_features = units
            
        # Capa de salida (26 letras en el abecedario)
        self.salida = nn.Linear(in_features, num_clases)

    def forward(self, x):
        # Pasar por convoluciones
        for layer in self.conv_layers:
            x = layer(x)
            
        # Aplanar de forma segura
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Pasar por capas densas
        for layer in self.dense_layers:
            x = layer(x)
            
        # Salida final
        x = self.salida(x)
        return x

# ==========================================
# 2. EARLY STOPPING
# ==========================================
class EarlyStopping:
    def __init__(self, paciencia=5, delta=0.0):
        self.paciencia = paciencia
        self.delta = delta
        self.mejor_loss = float('inf')
        self.contador = 0
        self.detener = False

    def __call__(self, val_loss):
        if val_loss < self.mejor_loss - self.delta:
            self.mejor_loss = val_loss
            self.contador = 0
        else:
            self.contador += 1
            if self.contador >= self.paciencia:
                self.detener = True

# ==========================================
# 3. BUCLE DE ENTRENAMIENTO
# ==========================================
def entrenar_modelo(cromosoma, dataloader_train, dataloader_val, epocas=20):
    """
    Entrena la CNN dinámica y retorna el mejor accuracy en validación.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = CNN_Dinamica(cromosoma).to(device)
    
    lr = cromosoma.get('lr', 0.001)
    
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)
    early_stopping = EarlyStopping(paciencia=3)
    
    mejor_val_acc = 0.0
    
    for epoca in range(epocas):
        # FASE DE ENTRENAMIENTO
        modelo.train()
        train_loss, correctos_train, total_train = 0.0, 0, 0
        
        for imagenes, etiquetas in dataloader_train:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            optimizador.zero_grad()
            salidas = modelo(imagenes)
            loss = criterio(salidas, etiquetas)
            loss.backward()
            optimizador.step()
            
            train_loss += loss.item()
            _, predichos = torch.max(salidas.data, 1)
            total_train += etiquetas.size(0)
            correctos_train += (predichos == etiquetas).sum().item()
            
        # FASE DE VALIDACIÓN
        modelo.eval()
        val_loss, correctos_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for imagenes, etiquetas in dataloader_val:
                imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
                salidas = modelo(imagenes)
                loss = criterio(salidas, etiquetas)
                
                val_loss += loss.item()
                _, predichos = torch.max(salidas.data, 1)
                total_val += etiquetas.size(0)
                correctos_val += (predichos == etiquetas).sum().item()
                
        # Calcular métricas finales de la época
        val_loss_promedio = val_loss / len(dataloader_val)
        val_acc = correctos_val / total_val
        
        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            
        print(f"Época {epoca+1}/{epocas} - Train Loss: {train_loss/len(dataloader_train):.4f} - Val Loss: {val_loss_promedio:.4f} - Val Acc: {val_acc:.4f}")
        
        # Verificar Early Stopping
        early_stopping(val_loss_promedio)
        if early_stopping.detener:
            print("Early stopping activado. Deteniendo entrenamiento.")
            break
    # Saber qué tan bueno fue el cromosoma
    return mejor_val_acc

# ==========================================
# 4. PRUEBA LOCAL
# ==========================================
if __name__ == "__main__":
    print("Probando el modelo con datos falsos (Dummy Data)...")
    
    # Cromosoma de prueba
    cromosoma_prueba = {
        'conv_filters': [16, 32], 
        'kernel_sizes': [3, 3], 
        'dense_units': [64],
        'lr': 0.005
    }
    
    imagenes_falsas = torch.randn(32, 1, 28, 28)
    # 32 etiquetas aleatorias entre 0 y 25
    etiquetas_falsas = torch.randint(0, 26, (32,))
    
    dataloader_falso = [(imagenes_falsas, etiquetas_falsas)]
    
    acc = entrenar_modelo(cromosoma_prueba, dataloader_train=dataloader_falso, dataloader_val=dataloader_falso, epocas=3)
    
    print(f"Prueba finalizada exitosamente. Mejor Accuracy Dummy: {acc:.4f}")