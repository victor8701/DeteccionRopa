# DeteccionRopa - Sistema Integrado

## 🚀 Uso Rápido

```bash
cd src
python3 main.py
```

Introduce el nombre de la imagen (sin extensión) cuando se solicite.

## ⚙️ Configuración

**Todos los parámetros están en `main.py` (líneas 16-68)**

### Parámetros Principales:
- `CONFIANZA_MINIMA = 0.5` - Umbral YOLO (↑ más preciso, ↓ más detecciones)
- `MARGEN_EXCLUSION_PRENDAS = 20` - Margen alrededor de prendas (↑ menos esquinas falsas)
- `MOSTRAR_VENTANA_YOLO = True` - Mostrar/ocultar ventanas
- `CARPETA_IMAGENES = "02Dic"` - Subcarpeta de imágenes

Ver comentarios en `main.py` para más detalles.

## 📁 Archivos Generados

- `runs/detect/predictX/detecciones.txt` - Reporte YOLO
- `resultados_deteccion/{imagen}_reporte.txt` - Reporte unificado con coordenadas

## 🔧 Estructura

- `main.py` - Script principal con configuración centralizada
- `deteccion_ropa.py` - Detección de prendas con YOLO
- `deteccion_esquinas.py` - Detección de esquinas de caja

## 📦 Dependencias

```bash
pip install ultralytics opencv-python
```

## 📥 Modelos YOLO (Descarga Requerida)

Los archivos de modelos YOLO (`.pt`) **no están incluidos** en el repositorio debido a su gran tamaño.

### Modelos Necesarios (en carpeta `src/`):
- `yolov8n.pt` (6.3 MB)
- `yolo11n.pt` (5.4 MB)
- `yolov8s-world.pt` (26 MB)
- `yolov8l-world.pt` (92 MB)
- `yolov8x-world.pt` (142 MB)

### Descarga Automática:
Los modelos se descargan automáticamente al ejecutar el código por primera vez si tienes `ultralytics` instalado.

### Descarga Manual (opcional):
```bash
cd src/
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-world.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-world.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

## 🎯 Clases Detectadas

bag, belt, boots, footwear, outer, dress, sunglasses, pants, top, shorts, skirt, headwear, scarf, tie
