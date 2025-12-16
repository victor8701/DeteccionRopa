# 📊 Análisis de Entrenamiento YOLO

Script para visualizar y analizar los resultados del entrenamiento de modelos YOLO.

---

## 🚀 Uso

```bash
cd src/analisis_entrenamiento
python3 analisis_entrenamiento.py
```

## 📦 Dependencias

```bash
pip install pandas matplotlib
```

---

## 📁 Archivos

| Archivo                       | Descripción                                   |
|-------------------------------|---------------------------------------------  |
| `analisis_entrenamiento.py`   | Script principal                              |
| `results.csv`                 | Datos del entrenamiento (exportado por YOLO)  |
| `grafico_perdidas.png`        | Gráfica de pérdidas (Box, Cls, DFL)           |
| `grafico_metricas.png`        | Gráfica de métricas (mAP, Precision, Recall)  |

---

## 📈 Gráficos Generados

### 1. Análisis de Pérdidas (Loss)

Compara el error en entrenamiento vs validación para detectar overfitting:

- **Box Loss:** Error en la localización de cajas
- **Cls Loss:** Error en la clasificación de objetos
- **DFL Loss:** Error en los bordes finos

**Interpretación:**
- Si la pérdida de validación sube mientras la de entrenamiento baja → **Overfitting**
- Si ambas bajan juntas → **Buen aprendizaje**

### 2. Métricas de Rendimiento

Muestra la evolución de la calidad del modelo:

- **mAP@50:** Precisión media con IoU ≥ 0.5 (más permisivo)
- **mAP@50-95:** Precisión media con IoU promediado (más estricto)
- **Precision:** Qué porcentaje de detecciones son correctas
- **Recall:** Qué porcentaje de objetos reales fueron detectados

**Interpretación:**
- mAP alto + Precision alta + Recall alto = **Buen modelo**
- Precision alta + Recall bajo = Detecta poco pero con certeza
- Precision baja + Recall alto = Detecta mucho pero con errores

---

## 📋 Columnas del CSV (results.csv)

| Columna                       | Descripción                                   |
|-------------------------------|---------------------------------------------  |
| `epoch`                       | Número de época                               |
| `train/box_loss`              | Pérdida de localización (entrenamiento)       |
| `train/cls_loss`              | Pérdida de clasificación (entrenamiento)      |
| `train/dfl_loss`              | Pérdida de bordes (entrenamiento)             |
| `val/box_loss`                | Pérdida de localización (validación)          |
| `val/cls_loss`                | Pérdida de clasificación (validación)         |
| `val/dfl_loss`                | Pérdida de bordes (validación)                |
| `metrics/precision(B)`        | Precisión en validación                       |
| `metrics/recall(B)`           | Recall en validación                          |
| `metrics/mAP50(B)`            | mAP con IoU ≥ 0.5                             |
| `metrics/mAP50-95(B)`         | mAP promedio (IoU 0.5-0.95)                   |

---

## 🔧 Cómo obtener results.csv

El archivo `results.csv` se genera automáticamente al entrenar un modelo YOLO:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='mi_dataset.yaml', epochs=100)
# El archivo results.csv se guarda en runs/detect/train/
```

Luego cópialo a esta carpeta:
```bash
cp runs/detect/train/results.csv src/analisis_entrenamiento/
```

---

## 👥 Autor

Proyecto de Visión por Computador - Máster en Robótica y Automática (UC3M)
