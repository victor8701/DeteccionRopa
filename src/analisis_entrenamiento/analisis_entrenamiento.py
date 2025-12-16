import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. CARGA Y LIMPIEZA DE DATOS
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'results.csv')

try:
    data = pd.read_csv(file_path)
    # IMPORTANTE: Los CSV de YOLO suelen tener espacios en los nombres de columnas.
    # Esto los elimina para evitar errores (ej: ' train/box_loss' -> 'train/box_loss')
    data.columns = [c.strip() for c in data.columns]
except FileNotFoundError:
    print(f"Error: No se encuentra el archivo {file_path}")
    exit()

# Configuración de estilo para gráficos bonitos
plt.style.use('bmh') # Estilo limpio y profesional

# 2. GRÁFICO 1: ANÁLISIS DE PÉRDIDAS (LOSS)
# -----------------------------------------
# Este gráfico compara el error en entrenamiento vs validación.
# Ideal para explicar si el modelo aprendió bien o memorizó.

fig_loss, axes_loss = plt.subplots(1, 3, figsize=(18, 5))
fig_loss.suptitle('Análisis de Aprendizaje: Pérdidas (Loss)', fontsize=16)

# Definimos las parejas de métricas a comparar
loss_metrics = [
    ('train/box_loss', 'val/box_loss', 'Box Loss (Cajas)'),
    ('train/cls_loss', 'val/cls_loss', 'Cls Loss (Clasificación)'),
    ('train/dfl_loss', 'val/dfl_loss', 'DFL Loss (Bordes Finos)')
]

for i, (train_col, val_col, title) in enumerate(loss_metrics):
    ax = axes_loss[i]
    ax.plot(data['epoch'].values, data[train_col].values, label='Entrenamiento', linewidth=2)
    ax.plot(data['epoch'].values, data[val_col].values, label='Validación', linewidth=2, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Pérdida')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'grafico_perdidas.png')) # Guarda la imagen
print("Gráfico de pérdidas guardado como 'grafico_perdidas.png'")

# 3. GRÁFICO 2: MÉTRICAS DE RENDIMIENTO (mAP, Precision, Recall)
# --------------------------------------------------------------
# Este gráfico muestra qué tan bueno es el modelo detectando.

fig_metrics, axes_metrics = plt.subplots(1, 2, figsize=(16, 6))
fig_metrics.suptitle('Evolución del Rendimiento del Modelo', fontsize=16)

# Subplot 1: mAP (Mean Average Precision) - La métrica más importante
ax1 = axes_metrics[0]
ax1.plot(data['epoch'].values, data['metrics/mAP50(B)'].values, label='mAP@50 (Holgado)', color='blue')
ax1.plot(data['epoch'].values, data['metrics/mAP50-95(B)'].values, label='mAP@50-95 (Estricto)', color='green', linewidth=2.5)
ax1.set_title('Precisión Media (mAP)')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Score (0-1)')
ax1.legend()
ax1.grid(True)

# Subplot 2: Precisión vs Recall
ax2 = axes_metrics[1]
ax2.plot(data['epoch'].values, data['metrics/precision(B)'].values, label='Precisión (Calidad)', color='purple', alpha=0.7)
ax2.plot(data['epoch'].values, data['metrics/recall(B)'].values, label='Recall (Cantidad)', color='orange', alpha=0.7)
ax2.set_title('Precisión vs. Recall')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Score (0-1)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'grafico_metricas.png')) # Guarda la imagen
print("Gráfico de métricas guardado como 'grafico_metricas.png'")

plt.show() # Muestra los gráficos en pantalla