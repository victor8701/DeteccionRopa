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

## 🎯 Clases Detectadas

bag, belt, boots, footwear, outer, dress, sunglasses, pants, top, shorts, skirt, headwear, scarf, tie
