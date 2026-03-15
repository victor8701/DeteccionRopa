# Solución Técnica: Adaptación del Sistema DetecciónRopa para Detección de Bolsas de Plástico Transparentes

**Fecha:** 2026-02-03  
**Versión:** 1.0  
**Autor:** Sistema de Análisis NotebookLM + DetecciónRopa

---

## 📋 Resumen Ejecutivo

Este documento proporciona una **hoja de ruta técnica completa** para adaptar tu sistema actual de detección de prendas de ropa (`DeteccionRopa`) a la **detección binaria de bolsas de plástico transparentes apiladas** ("bultos").

**Cambio principal:** De clasificación multi-clase (zapatos, vestidos, camisas) a **detección binaria objeto/no-objeto** (bulto/fondo).

**Desafíos específicos:**
- Transparencia y reflejos especulares
- Deformabilidad al agarrar
- Apilamientos con oclusiones parciales
- Ausencia de textura visual
- Variaciones en iluminación

**Estrategia recomendada:** Migrar a **RGE-YOLO** (modificación de YOLOv8s optimizada para bolsas deformables) con pipeline de preprocesamiento especializado y técnica de detección de bordes Sobel como preprocesamiento complementario.

---

## 1. Análisis del Código Actual

### 1.1 Componentes Existentes

Tu sistema actual consta de **3 módulos** integrados:

#### **A) `deteccion_ropa.py` - Detección YOLO Multi-Clase**

**Función principal:** `detectar_ropa()`

**Características:**
- Usa modelo `runs/detect/train/weights/best.pt` (YOLOv8 entrenado)
- Clasificación multi-clase (belt, boots, jacket, dress, etc.)
- Parámetros de inferencia: `conf=0.1`, `imgsz=1280`, `augment=True`, `iou=0.5`
- Lógica especial para **pares** (zapatos): Agrupa detecciones cercanas de calzado
- Filtrado de duplicados usando IoU y proximidad de centros

**Fortalezas para adaptar:**
- ✅ `imgsz=1280`: Excelente para objetos transparentes (preserva bordes sutiles)
- ✅ `augment=True`: Test-Time Augmentation ayuda con variabilidad
- ✅ Sistema de filtrado robusto

**Debilidades:**
- ❌ `conf=0.1`: Demasiado bajo para inferencia productiva (compensa falta de precisión)
- ❌ Lógica de pares innecesaria para bultos
- ❌ Modelo multi-clase cuando necesitas binario

#### **B) `deteccion_esquinas.py` - Detección de Bordes y Esquinas**

**Funciones clave:**
1. `border_sobel()`: Aplica filtro Sobel para detectar bordes
2. `obtener_mascara_carton_filtrada()`: Máscara HSV + morfología para aislar caja de cartón
3. `detectar_esquinas_caja()`: Shi-Tomasi (goodFeaturesToTrack) con máscara de exclusión

**Pipeline actual:**
```
Imagen RGB → Grayscale → Sobel X+Y → Máscara HSV (color cartón) → 
Morfología (CLOSE) → Contorno de caja → Máscara ROI → 
Exclusión de prendas → goodFeaturesToTrack → Esquinas
```

**Relevancia para bolsas plásticas:**
- ✅ **Sobel es CRÍTICO**: Los objetos transparentes solo tienen bordes visibles (sin textura)
- ✅ Máscara de exclusión: Útil para aislar zonas de interés
- ⚠️ HSV para cartón no aplicable directamente a plástico transparente

#### **C) `main.py` - Orquestador

Coordina ambos módulos:
1. Ejecuta YOLO → obtiene bboxes de prendas
2. Pasa bboxes a `deteccion_esquinas` como máscara de exclusión
3. Dibuja esquinas rojas + centros de prendas verdes

---

### 1.2 Evaluación de Parámetros Actuales

**`imgsz = 1280`**  
✅ **MANTENER** - Bordes de plástico necesitan alta resolución

**`augment = True`**  
✅ **MANTENER** - Esencial para transparencias

**`conf = 0.1`**  
❌ Muy bajo - **SUBIR a 0.25** después de reentrenar

**`iou` (NMS) `= 0.5`**  
⚠️ Estándar - Evaluar en entrenamiento

**Lógica de pares**  
❌ Innecesaria - **ELIMINAR** completamente

---

## 2. Estrategia de Adaptación Recomendada

### 2.1 Arquitectura: ¿RGE-YOLO o YOLOv12?

Según las fuentes de NotebookLM, hay dos opciones principales:

#### **Opción A: RGE-YOLO** ⭐ **RECOMENDADA**

**Ventajas específicas para bolsas deformables:**
1. **Atención EMA (Efficient Multi-Scale Attention):** Captura información espacial global en lugar de texturas locales. Crítico cuando las bolsas se arrugan y pierden forma canónica.
2. **Bloques RepViT:** Reduce parámetros manteniendo precisión. Permite 250 FPS en hardware limitado.
3. **GSConv (Ghost Shuffle Convolutions):** Reduce costo computacional sin perder detalles de borden.
4. **Diseñado explícitamente para bolsas deformables** en carreteras (paper 2025).

**Desventajas:**
- Requiere modificación manual del archivo `.yaml` de YOLOv8
- Menos soporte oficial que YOLOv12

**Casos de uso ideales:**
- ✅ Detección de presencia y ubicación del bulto para agarre robótico
- ✅ Hardware limitado (Jetson Nano, Xavier NX)
- ✅ Formas aleatorias/arrugadas

---

#### **Opción B: YOLOv12**

**Ventajas:**
1. **Flash Attention:** Superior para desambiguar objetos superpuestos en entornos de "basura mixta"
2. **r-ELAN:** Mejor para texturas complejas
3. **Precisión mAP@0.5 ~88% en residuos** (vs 85% de RGE-YOLO)

**Desventajas:**
- Requiere GPU potente (NVIDIA Tesla/RTX)
- ~25.5M parámetros (vs 9.1M de RGE-YOLO)
- Más lento (~200 FPS vs 250 FPS)

**Casos de uso ideales:**
- ✅ Distinguir bolsas de otros tipos de basura
- ✅ Apilamientos extremadamente densos
- ✅ GPU de alta gama disponible

---

#### **Decisión Recomendada: RGE-YOLO**

**Razón:** Tu problema principal es la **deformabilidad** de la bolsa en sí, no distinguirla de otros objetos. RGE-YOLO fue diseñado para resolver exactamente esto (bolsas irregulares en la carretera). Además, puedes seguir usando Ultralytics YOLOv8 editando el `.yaml` en lugar de cambiar de framework.

---

### 2.2 Modificaciones al Modelo

#### **Paso 1: Preparar Dataset Binario**

```bash
# Estructura de carpeta
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

**Formato de label (YOLO):**
```
0 x_center y_center width height
```
- **Clase 0:** `bulto` (única clase)
- Coordenadas normalizadas (0-1)

**Requisitos de anotación:**
- La bounding box debe englobar **todo el plástico visible**, incluso si contiene múltiples objetos internos
- No etiquetar objetos parcialmente visibles (≤30% visible)

---

#### **Paso 2: Modificar Arquitectura a RGE-YOLO**

**Archivo a editar:** `yolov8.yaml` (copia del modelo base)

**Cambios necesarios:**
1. **Reemplazar bloques C2f por RepViT** en el backbone
2. **Añadir módulo EMA** al final del backbone
3. **Integrar GSConv** en el neck

**Referencia:**  
No puedo proporcionarte el código `.yaml` exacto sin acceso al paper completo de RGE-YOLO, pero la estructura general sería:

```yaml
# Modified RGE-YOLO based on YOLOv8s

backbone:
  # Head
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, RepViT, [128]]      # 2 (Reemplazar C2f)
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, RepViT, [256]]      # 4 (Reemplazar C2f)
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, RepViT, [512]]      # 6 (Reemplazar C2f)
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, RepViT, [1024]]     # 8 (Reemplazar C2f)
  - [-1, 1, EMA, [1024]]        # 9 (NUEVO - Módulo de atención)
  - [-1, 1, SPPF, [1024, 5]]    # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, GSConv, [512]]     # 13 (Reemplazar C2f)

  # ... (continuar con el resto del head usando GSConv)
```

**Nota:** Si no tienes acceso al código RepViT/EMA/GSConv, puedes empezar con YOLOv8s estándar y evaluar si la mejora justifica la complejidad.

---

#### **Paso 3 (Alternativa Rápida): Transfer Learning con YOLOv8s Estándar**

Si deseas probar rápido sin modificar la arquitectura:

```python
from ultralytics import YOLO

# Usar YOLOv8s preentrenado en COCO
model = YOLO('yolov8s.pt')

# Entrenar con tu dataset binario
results = model.train(
    data='bolsas.yaml',   # Tu archivo de configuración
    epochs=100,
    imgsz=1280,           # MANTENER alta resolución
    batch=8,              # Ajustar según GPU
    device=0,             # GPU
    project='runs/detect',
    name='bolsas_plasticas_v1',
    
    # Hiperparámetros específicos (ver sección 4.3)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1
)

# Tu nuevo best.pt estará en runs/detect/bolsas_plasticas_v1/weights/best.pt
```

---

### 2.3 Actualizar Parámetros de Inferencia

Modificar `deteccion_ropa.py`:

```python
# ANTES
results_raw = model.predict(
    source=ruta_imagen, 
    save=False, 
    conf=0.1,        # ❌ Muy bajo
    imgsz=1280,      # ✅ Mantener
    augment=True,    # ✅ Mantener
    iou=0.5          # ⚠️ Evaluar
)

# DESPUÉS (para bolsas)
results_raw = model.predict(
    source=ruta_imagen, 
    save=False, 
    conf=0.25,       # ✅ Subir a 0.25 (según fuentes)
    imgsz=1280,      # ✅ Mantener (crítico para bordes)
    augment=True,    # ✅ Mantener (TTA para transparencias)
    iou=0.45         # ⚠️ Ajustar en validación
)
```

**Justificación `conf=0.25`:** Las fuentes recomiendan este valor para objetos con baja textura después de un entrenamiento adecuado.

---

## 3. Pipeline de Preprocesamiento

### 3.1 ¿Por Qué Preprocesar?

Los objetos transparentes tienen **características visuales únicas**:
- **NO tienen textura interna** (el patrón visible es el fondo detrás del plástico)
- **SÍ tienen bordes nítidos** (refracción de la luz en los límites del material)
- **Reflejos especulares** (puntos brillantes que confunden a YOLO)

**Solución:** Convertir a **grayscale** para eliminar confusión de color y enfatizar bordes vía contraste.

---

### 3.2 Pipeline Recomendado (Inferencia)

Añadir **ANTES** de `model.predict()`:

```python
import cv2
import numpy as np

def preprocesar_para_bolsas(imagen_bgr):
    """
    Preprocesa imagen para mejorar detección de bordes transparentes.
    
    Args:
        imagen_bgr: Imagen original en formato BGR
    
    Returns:
        imagen_procesada: Imagen en grayscale lista para YOLO
    """
    # Paso 1: Convertir a grayscale
    # Razón: Elimina confusión de color, enfatiza bordes por contraste
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    
    # Paso 2: Ecualización de histograma (opcional)
    # Útil si hay variaciones extremas de iluminación
    # gray = cv2.equalizeHist(gray)
    
    # Paso 3: Convertir de vuelta a 3 canales para YOLO
    # YOLO espera input de 3 canales
    imagen_procesada = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return imagen_procesada

# USO
imagen_original = cv2.imread(ruta_imagen)
imagen_prep = preprocesar_para_bolsas(imagen_original)

# Ahora pasar a YOLO
results = model.predict(source=imagen_prep, ...)
```

**Nota:** Este preprocesamiento es **SOLO para inferencia**. No lo uses en entrenamiento (el data augmentation debe aplicarse sobre imágenes RGB normales).

---

### 3.3 Preprocesamiento Avanzado (Opcional)

Si las bolsas tienen **motion blur** por movimiento de cinta transportadora:

```python
def preprocesar_avanzado(imagen_bgr):
    """
    Preprocesamiento con reducción de motion blur.
    """
    # Grayscale
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur ligero para reducir ruido
    # kernel=3 (No más grande o perderás bordes)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Sharpening via Canny + OR con imagen original
    # (Re-enfatiza bordes que se borraron)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Combinar: Imagen suavizada + bordes detectados
    sharpened = cv2.addWeighted(blurred, 0.7, edges, 0.3, 0)
    
    # Volver a BGR
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
```

**Advertencia:** Esto puede ser contraproducente si el motion blur es sintético (data augmentation). Prueba A/B.

---

### 3.4 Integración con Sobel (Detección de Esquinas)

Tu módulo `deteccion_esquinas.py` **ya detecta bordes con Sobel**. Puedes **reutilizar esta lógica** como preprocesamiento alternativo:

```python
def preprocesar_con_sobel(imagen_bgr):
    """
    Usa Sobel para enfatizar bordes antes de YOLO.
    """
    # Grayscale
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    
    # Sobel X + Y
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # Combinar con imagen original (70% original + 30% Sobel)
    combined = cv2.addWeighted(gray, 0.7, sobel, 0.3, 0)
    
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
```

**Ventaja:** Sobel ya detecta los límites transparentes en tu código actual. Integrarlo en YOLO podría mejorar detección.

**Desventaja:** YOLO aprende bordes automáticamente. Esto puede ser redundante.

**Recomendación:** Prueba en validación:
1. Solo grayscale
2. Grayscale + Sobel (30% peso)
3. Sin preprocesamiento

Mide mAP@0.5 y elige el mejor.

---

## 4. Data Augmentation para Entrenamiento

### 4.1 Importancia Crítica

Las bolsas plásticas presentan **variabilidad extrema**:
- Deformaciones al agarrar
- Transparencia variable (grosor del plástico)
- Reflejos especulares aleatorios
- Motion blur por movimiento

**Solución:** Data augmentation agresivo durante entrenamiento para que el modelo aprenda invariancia.

---

### 4.2 Configuración Recomendada (hyp.scratch.yaml)

Crear/editar `hyp.scratch-bolsas.yaml`:

```yaml
# =============================================================================
# HIPERPARÁMETROS PARA BOLSAS DE PLÁSTICO TRANSPARENTES
# Basado en fuentes NotebookLM: RGE-YOLO paper + YOLOv12 residuos
# =============================================================================

# --- Color/Brightness (Bajo impacto en transparentes) ---
hsv_h: 0.015    # Hue variation ±1.5% (color importa poco)
hsv_s: 0.7      # Saturation ±70% (simula diferentes grosores de plástico)
hsv_v: 0.4      # Value/Brightness ±40% (variaciones de iluminación)

# --- Transformaciones Geométricas ---
degrees: 15.0       # Rotation ±15° (bolsas en cualquier orientación)
translate: 0.1      # Translation ±10% (posiciones aleatorias en cinta)
scale: 0.5          # Scale 50%-150% (bolsas grandes y pequeñas)
shear: 0.0          # Shear desactivado (deformación se simula con motion blur)
perspective: 0.0    # Perspective desactivado

# --- Flips ---
flipud: 0.0    # Vertical flip desactivado (gravedad importa)
fliplr: 0.5    # Horizontal flip 50% (simetría)

# --- Advanced Augmentations ---
mosaic: 1.0        # Mosaic SIEMPRE (simula apilamiento)
mixup: 0.1         # Mixup 10% (transparencias superpuestas)
copy_paste: 0.1    # Copy-Paste 10% (ver sección 4.3)

# --- Motion Blur (MUY IMPORTANTE) ---
# Esto se añade via código personalizado (ver sección 4.3.1)

# --- Otros ---
degrees: 0.0       # No rotación extra en Mosaic
translate: 0.1
scale: 0.5
```

---

### 4.3 Augmentations Críticos Específicos

#### **4.3.1 Linear Motion Blur** ⭐ **PRIORIDAD MÁXIMA**

Las bolsas en movimiento pierden textura. Debes entrenar con desenfoque.

**Implementación con Albumentations:**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    # Motion Blur Lineal
    A.MotionBlur(
        blur_limit=(20, 40),      # Kernel size 20-40 píxeles
        allow_shifted=False,
        p=0.5                      # Aplicar en 50% de imágenes
    ),
    
    # Gaussian Blur (ruido de cámara)
    A.GaussianBlur(
        blur_limit=(3, 7),
        sigma_limit=0,
        p=0.3
    ),
    
    # Variaciones de brillo/contraste
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    
    # Simulación de reflejos especulares
    A.RandomSunFlare(
        flare_roi=(0, 0, 1, 0.5),  # Mitad superior de imagen
        angle_lower=0,
        angle_upper=1,
        num_flare_circles_lower=1,
        num_flare_circles_upper=2,
        src_radius=50,
        p=0.1                       # Bajo (reflejos son raros)
    ),
    
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

**IMPORTANTE:** Aplica motion blur **también al canal alpha de la máscara de segmentación** si usas segmentación instance.

---

#### **4.3.2 Copy-Paste para Apilamiento**

Simula bolsas apiladas pegando máscaras sobre fondos aleatorios.

**Parámetro clave:** IoU threshold = 0.1

```python
# Pseudo-código de lógica
def copy_paste_bags(image, masks, max_bags=10):
    for i in range(random.randint(2, max_bags)):
        # Seleccionar máscara aleatoria
        mask = random.choice(masks)
        
        # Posición aleatoria
        x, y = random_position()
        
        # Verificar IoU con bolsas ya colocadas
        iou = calculate_iou(mask, existing_masks, x, y)
        
        if iou < 0.1:  # Superposición ligera permitida
            paste_mask(image, mask, x, y)
            existing_masks.append((mask, x, y))
    
    return image
```

**Resultado:** Dataset sintético con 2-10 bolsas por imagen, algunas tocándose (IoU < 0.1) pero no completamente superpuestas.

---

#### **4.3.3 Simulación de Cáusticas (Datos Sintéticos)**

Si generas datos sintéticos con Blender:

**Motor de renderizado:** LuxCoreRender (NO Eevee/Cycles estándar)

**Razón:** LuxCore simula correctamente la refracción de luz a través del plástico (cáusticas). Los motores de juegos crean plástico "falso" que YOLO aprenderá y fallará en escenas reales.

---

### 4.4 Ejemplo Completo de Pipeline de Entrenamiento

```python
from ultralytics import YOLO
import albumentations as A

# Cargar modelo base
model = YOLO('yolov8s.pt')  # O tu yolov8-rge.yaml personalizado

# Entrenar
results = model.train(
    data='bolsas.yaml',
    epochs=100,
    imgsz=1280,                  # Alta resolución
    batch=8,
    device=0,
    
    # Hiperparámetros de augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    
    # Callbacks personalizados para motion blur
    # (requiere implementación via Trainer callbacks)
    
    # Otras opciones
    patience=20,                 # Early stopping
    save=True,
    save_period=10,              # Guardar checkpoint cada 10 epochs
    cache=True,                  # Cachear dataset en RAM si cabe
    workers=8,
    project='runs/bolsas',
    name='rge_yolo_v1'
)
```

---

## 5. Cambios en el Código Python

### 5.1 Eliminar Lógica de Pares

En `deteccion_ropa.py`, **ELIMINAR completamente** la función `filtrar_duplicados()` y su lógica de agrupación de zapatos.

**Justificación:** La robótica moderna usa **Soft Grippers** (pinzas blandas neumáticas) que se adaptan mecánicamente a la forma del objeto. No necesitas separar objetos internos del bulto.

**Antes:**
```python
def filtrar_duplicados(boxes, clases, confs, nombres_clases, iou_threshold=0.3):
    # ... Lógica compleja para pares de zapatos ...
    pass
```

**Después:**
```python
# Función eliminada. Usar directamente NMS de YOLO con iou_threshold adecuado.
# YOLO ya hace Non-Maximum Suppression internamente.
```

**Cambio en flujo:**

```python
# ANTES
detecciones_raw = model.predict(...)
detecciones_filtradas = filtrar_duplicados(...)

# DESPUÉS
detecciones = model.predict(..., iou=0.45)  # NMS integrado
# Usar directamente las detecciones
```

---

### 5.2 Cambiar Modelo a Binario

```python
# ANTES
path_modelo = 'runs/detect/train/weights/best.pt'  # Multi-clase
model = YOLO(path_modelo)
# model.set_classes(mis_clases)  # Lista de clases múltiples

# DESPUÉS
path_modelo = 'runs/bolsas/rge_yolo_v1/weights/best.pt'  # Binario
model = YOLO(path_modelo)
# No necesitas set_classes (solo hay clase 0: 'bulto')
```

---

### 5.3 Añadir Preprocesamiento

```python
def detectar_bolsas(ruta_imagen, mostrar_ventana=True, conf_min=0.25):
    """
    Detecta bultos (bolsas de plástico) en imagen.
    
    Returns:
        Lista de tuplas: [(centro_x, centro_y, x1, y1, x2, y2, confianza), ...]
    """
    # Cargar modelo
    model = YOLO('runs/bolsas/rge_yolo_v1/weights/best.pt')
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    
    # NUEVO: Preprocesamiento
    imagen_prep = preprocesar_para_bolsas(imagen)
    
    # Predicción
    results = model.predict(
        source=imagen_prep,
        save=False,
        conf=conf_min,
        imgsz=1280,
        augment=True,
        iou=0.45
    )
    
    # Extraer detecciones
    bultos = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            # Centro
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            bultos.append((cx, cy, int(x1), int(y1), int(x2), int(y2), float(conf)))
    
    return bultos
```

---

### 5.4 Integración con Detección de Zonas (QR Codes)

Según las fuentes, para sistemas de robótica, es común usar **códigos QR** como referencias espaciales para estimar profundidad y delimitar zonas de trabajo.

**Nuevo flujo recomendado:**

```
1. Detectar QR codes en imagen → Calcular transformación de perspectiva
2. Definir Zonas de Interés (ROIs) relativas a QR
3. Detectar bultos con YOLO
4. Filtrar bultos que estén dentro de ROIs válidas
5. Ordenar por prioridad (ej: zona 1 > zona 2 > zona 3)
6. Enviar coordenadas de bulto prioritario al brazo robótico
```

**Ejemplo básico:**

```python
import cv2
from pyzbar import pyzbar

def detectar_zonas_qr(imagen):
    """
    Detecta códigos QR y define zonas de trabajo.
    
    Returns:
        zonas: Lista de ROIs [(x1, y1, x2, y2, prioridad), ...]
    """
    # Detectar QR codes
    qr_codes = pyzbar.decode(imagen)
    
    zonas = []
    for i, qr in enumerate(qr_codes):
        # Extraer esquinas del QR
        points = qr.polygon
        
        # Definir ROI centrada en QR (ej: 500x500 píxeles)
        cx = sum([p.x for p in points]) // len(points)
        cy = sum([p.y for p in points]) // len(points)
        
        x1 = cx - 250
        y1 = cy - 250
        x2 = cx + 250
        y2 = cy + 250
        
        # Prioridad según data del QR (ej: "ZONA_1", "ZONA_2")
        prioridad = int(qr.data.decode('utf-8').split('_')[1])
        
        zonas.append((x1, y1, x2, y2, prioridad))
    
    return zonas

def filtrar_bultos_por_zona(bultos, zonas):
    """
    Filtra bultos que estén en zonas válidas y los ordena por prioridad.
    """
    bultos_validos = []
    
    for bulto in bultos:
        cx, cy, x1, y1, x2, y2, conf = bulto
        
        for zx1, zy1, zx2, zy2, prioridad in zonas:
            # Verificar si centro del bulto está en zona
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                bultos_validos.append((bulto, prioridad))
                break
    
    # Ordenar por prioridad (menor número = mayor prioridad)
    bultos_validos.sort(key=lambda x: x[1])
    
    return [b[0] for b in bultos_validos]  # Retornar solo bultos
```

**Uso:**

```python
# 1. Detectar zonas
imagen = cv2.imread(ruta)
zonas = detectar_zonas_qr(imagen)

# 2. Detectar bultos
bultos = detectar_bolsas(ruta, conf_min=0.25)

# 3. Filtrar por zona
bultos_validos = filtrar_bultos_por_zona(bultos, zonas)

# 4. Enviar primer bulto válido al robot
if bultos_validos:
    bulto_target = bultos_validos[0]
    enviar_a_robot(bulto_target)
```

---

## 6. Recomendaciones de Hardware/Robótica

### 6.1 Soft Grippers Recomendados

Según las fuentes, para **bolsas deformables** se recomiendan:

#### **Opción 1: Origami Magic Ball Gripper**
- Adaptación pasiva a geometría irregular
- Agarre envolvente distribuyencio presión
- Ideal para formas impredecibles

#### **Opción 2: Dedos Neumáticos (Pneumatic Fingers)**
- Bajo costo
- Robustos a variaciones de forma
- Requieren compresor de aire

**Ventaja clave:** NO necesitas conocer la forma exacta interna del bulto. El gripper se deforma para adaptarse.

---

### 6.2 Hardware de Cómputo

**Para RGE-YOLO:**
- ✅ NVIDIA Jetson Xavier NX (16GB RAM) - Recomendado
- ✅ NVIDIA Jetson Orin Nano
- ⚠️ Jetson Nano (4GB) - Limitado, reducir batch size

**Para YOLOv12:**
- ✅ NVIDIA RTX 3070 o superior
- ✅ NVIDIA Tesla T4 (cloud)

**FPS esperado (imgsz=1280):**
- RGE-YOLO en Xavier NX: ~30-40 FPS
- YOLOv12 en RTX 3080: ~50-60 FPS

---

## 7. Plan de Implementación Paso a Paso

### **Fase 1: Preparación (Semana 1)**

- [ ] Recopilar dataset de bolsas plásticas (mínimo 500 imágenes)
- [ ] Anotar con LabelImg/Roboflow (clase única: `bulto`)
- [ ] Dividir: 70% train, 20% val, 10% test
- [ ] Crear `bolsas.yaml`:
```yaml
path: /ruta/a/dataset/bolsas
train: images/train
val: images/val
test: images/test

nc: 1  # Número de clases
names: ['bulto']  # Nombres de clases
```

---

### **Fase 2: Entrenamiento Inicial (Semana 2)**

- [ ] Entrenar YOLOv8s estándar como baseline:
```bash
yolo detect train data=bolsas.yaml model=yolov8s.pt epochs=100 imgsz=1280 batch=8
```
- [ ] Evaluar mAP@0.5 en validation set
- [ ] Si mAP < 70%, aumentar epochs o revisar anotaciones

---

### **Fase 3: Optimización (Semana 3)**

- [ ] Implementar data augmentation con Albumentations (motion blur, copy-paste)
- [ ] Re-entrenar con augmentation agresivo
- [ ] Probar preprocesamiento (grayscale, Sobel)
- [ ] Comparar mAP de cada variante

---

### **Fase 4: Migración a RGE-YOLO (Semana 4 - Opcional)**

- [ ] Modificar `yolov8.yaml` para integrar RepViT + EMA
- [ ] Re-entrenar con arquitectura modificada
- [ ] Comparar precisión vs YOLOv8s estándar
- [ ] Si mejora < 3%, mantener YOLOv8s (más simple)

---

### **Fase 5: Integración (Semana 5)**

- [ ] Actualizar `deteccion_ropa.py` → `deteccion_bultos.py`
- [ ] Eliminar lógica de pares
- [ ] Añadir preprocesamiento seleccionado
- [ ] Implementar detección de zonas con QR codes (si aplica)
- [ ] Integrar con `deteccion_esquinas.py` para referencia espacial

---

### **Fase 6: Validación en Campo (Semana 6)**

- [ ] Probar en escenario real con cinta transportadora
- [ ] Medir:
  - Precisión de detección (TP, FP, FN)
  - Tasa de agarre exitoso del robot
  - FPS en hardware objetivo
- [ ] Ajustar `conf_min` según balance precisión/recall
- [ ] Iterar augmentation si hay casos de fallo recurrentes

---

## 8. Métricas de Éxito

| Métrica | Target | Método de Medición |
|---------|--------|-------------------|
| **mAP@0.5** | ≥ 85% | Validation set |
| **Precision** | ≥ 90% | TP / (TP + FP) |
| **Recall** | ≥ 80% | TP / (TP + FN) |
| **FPS** | ≥ 25 | Tiempo real en hardware |
| **Tasa de Agarre** | ≥ 85% | Pruebas robóticas |

---

## 9. Troubleshooting

### **Problema 1: mAP bajo (< 70%)**

**Posibles causas:**
- Dataset pequeño (< 500 imágenes)
- Anotaciones incorrectas (bboxes no ajustadas)
- Falta de variabilidad en imágenes (todas con misma iluminación)

**Soluciones:**
1. Aumentar dataset (mínimo 1000 imágenes)
2. Revisar anotaciones manualmente
3. Añadir augmentation más agresivo
4. Usar transfer learning desde dataset de residuos plásticos público

---

### **Problema 2: Muchos falsos positivos (reflejos, sombras)**

**Causa:** Modelo confunde reflejos especulares con bordes de bolsa

**Soluciones:**
1. Añadir imágenes negativas (sin bolsas) al dataset
2. Aumentar `conf_min` a 0.3-0.4
3. Post-procesamiento: Filtrar detecciones muy pequeñas (área < 1000 px²)
4. Entrenar con augmentation de RandomSunFlare para aprender reflejos como ruido

---

### **Problema 3: No detecta bolsas apiladas**

**Causa:** Modelo no vio suficientes ejemplos de apilamiento en entrenamiento

**Soluciones:**
1. Incrementar `copy_paste=0.3` en augmentation
2. Reducir `iou_threshold` en NMS a 0.3 (permitir más overlap)
3. Anotar manualmente casos de apilamiento extremo

---

### **Problema 4: FPS muy bajo (< 10)**

**Causas:**
- `imgsz` muy alto para hardware
- Modelo muy pesado (YOLOv12 en Jetson Nano)

**Soluciones:**
1. Reducir `imgsz` a 960 o 640 (medir pérdida de precisión)
2. Usar YOLOv8n (nano) en lugar de YOLOv8s
3. Quantización del modelo (TensorRT, ONNX INT8)
4. Actualizar hardware a Jetson Xavier NX

---

## 10. Recursos y Referencias

### **Papers Citados (de NotebookLM):**
1. RGE-YOLO: Detección de bolsas deformables (2025)
2. YOLOv12: Clasificación de residuos en tiempo real
3. Synthetic Data Generation for Transparent Objects
4. Soft Grippers for Deformable Objects (Origami Magic Ball)

### **Datasets Públicos Recomendados:**
- **TACO Dataset:** Basura en entornos naturales (incluye plásticos)
- **TrashNet:** Clasificación de residuos (6 clases, incluye plástico)
- **PlasticNet:** Específico para plásticos (transparentes y opacos)

### **Herramientas:**
- **Roboflow:** Anotación + augmentation automático
- **Albumentations:** Augmentation avanzado (motion blur, cáusticas)
- **LabelImg:** Anotación manual rápida
- **pyzbar:** Detección de QR codes para zonas

---

## 11. Conclusión y Próximos Pasos

### **Resumen de Cambios Clave:**

| Componente | Actual | Nuevo |
|------------|--------|-------|
| **Modelo** | YOLOv8 multi-clase | RGE-YOLO binario |
| **Clases** | 14 prendas | 1 clase (bulto) |
| **Preprocesamiento** | Ninguno | Grayscale + opcional Sobel |
| **Confidence** | 0.1 | 0.25 |
| **Lógica de pares** | Sí | NO (eliminada) |
| **Detección de zonas** | No | Sí (QR codes) |
| **Augmentation** | Básico | Motion blur + Copy-Paste |

---

### **Siguientes Acciones:**

1. **Inmediatas (Esta semana):**
   - Recopilar 100 imágenes de bolsas plásticas para prueba
   - Anotarlas con clase única `bulto`
   - Entrenar YOLOv8s baseline

2. **Corto plazo (2 semanas):**
   - Implementar augmentation completo
   - Evaluar preprocesamiento Sobel
   - Alcanzar mAP@0.5 ≥ 80%

3. **Largo plazo (1 mes):**
   - Migrar a RGE-YOLO si justifica la complejidad
   - Integrar con sistema robótico
   - Validar en producción

---

---

## 📞 Soporte

Si necesitas ayuda adicional:
1. Consulta las fuentes de NotebookLM para detalles técnicos
2. Revisa issues de Ultralytics YOLO en GitHub
3. Contacta con el equipo de robótica para pruebas de integración

---

## Apéndice: Estrategia de Podcasts en NotebookLM

### 🎙️ Recomendación: 3 Podcasts Temáticos

Para maximizar el aprendizaje profundo en cada área técnica, se recomienda generar **3 audio overviews especializados** en NotebookLM usando las 18 fuentes disponibles.

---

### **Podcast 1: "Fundamentos y Arquitecturas YOLO"** (15-20 min)

**Fuentes a seleccionar:**
- ✅ Documentación Ultralytics YOLOv8 (Explore Ultralytics YOL...)
- ✅ Papers sobre RGE-YOLO para bolsas deformables
- ✅ Papers sobre YOLOv12 para clasificación de residuos
- ✅ GitHub repositories con implementaciones de arquitecturas
- ✅ Secciones 1-2 de este documento (Análisis actual + Estrategia de adaptación)

**Contenido esperado:**
- Comparación técnica YOLOv8 vs RGE-YOLO vs YOLOv12
- Módulos especializados: EMA (Efficient Multi-Scale Attention), RepViT, GSConv
- Ventajas de cada arquitectura para objetos deformables y transparentes
- Decisión técnica: ¿Cuál implementar primero según hardware disponible?
- Justificación de detección binaria vs multi-clase

**Cuándo escucharlo:** 
Antes de empezar a entrenar, para tomar decisión informada sobre arquitectura base.

---

### **Podcast 2: "Detección de Objetos Transparentes y Pipeline Técnico"** (15-20 min)

**Fuentes a seleccionar:**
- ✅ Frontiers | Vision-based transparent objects (desafíos ópticos)
- ✅ Papers sobre preprocesamiento de imágenes
- ✅ Documentación de data augmentation
- ✅ Notas guardadas sobre: transformaciones OpenCV, reducción de reflejos, motion blur
- ✅ Secciones 3-4 de este documento (Pipeline de Preprocesamiento + Data Augmentation)

**Contenido esperado:**
- Características visuales únicas de objetos transparentes (sin textura, solo bordes)
- Pipeline de preprocesamiento: grayscale, Sobel, Canny edge detection
- Data augmentation crítico:
  - Linear Motion Blur (kernel 20-40px, 50% aplicación)
  - Copy-Paste para apilamiento (IoU threshold 0.1)
  - Simulación de reflejos especulares y cáusticas
- Parámetros específicos del archivo `hyp.scratch.yaml`
- Integración con detección de esquinas existente (Sobel + Shi-Tomasi)

**Cuándo escucharlo:** 
Mientras preparas el dataset y configuras el pipeline de augmentation con Albumentations.

---

### **Podcast 3: "Implementación Práctica y Robótica"** (15-20 min)

**Fuentes a seleccionar:**
- ✅ Introduction.pdf (contexto del proyecto industrial)
- ✅ Papers sobre Soft Grippers robóticos
- ✅ Notas sobre QR codes para detección de zonas
- ✅ Common Industrial Applications 
- ✅ Secciones 5-7 de este documento (Cambios en Código + Hardware + Plan de Implementación)

**Contenido esperado:**
- Modificaciones específicas a `deteccion_ropa.py`:
  - Eliminación de función `filtrar_duplicados()` y lógica de pares
  - Cambio de modelo multi-clase a binario
  - Actualización de parámetros de inferencia (conf=0.25, iou=0.45)
- Integración con sistema robótico:
  - Soft Grippers recomendados: Origami Magic Ball vs Pneumatic Fingers
  - Detección de zonas con QR codes (biblioteca pyzbar)
  - Pipeline completo: QR → ROI → YOLO → Filtrado → Priorización → Robot
- Hardware de deploy: Jetson Xavier NX vs RTX 3070
- Plan de implementación en 6 fases (semana a semana)
- Métricas de éxito y troubleshooting común

**Cuándo escucharlo:** 
Mientras modificas el código Python y planificas la integración con el brazo robótico.

---

### 📋 Proceso Recomendado

**Genera los podcasts en este orden:**

1. **Podcast 1** → Escucha → Decide arquitectura (RGE-YOLO vs YOLOv8s)
2. **Podcast 2** → Escucha → Prepara dataset con augmentation correcto
3. **Podcast 3** → Escucha → Implementa cambios en código e integración robótica

**Ventajas de 3 podcasts temáticos:**
- 🎯 **Enfoque específico:** Cada podcast profundiza en un área sin distracciones
- 📚 **Aprendizaje incremental:** Evolución lógica desde concepto → técnica → implementación
- ⏸️ **Pausas naturales:** Puedes escuchar uno, implementar lo aprendido, y continuar
- 🔄 **Re-escucha selectiva:** Si necesitas repasar augmentation, solo re-escuchas Podcast 2

**Generación en NotebookLM:**
1. Asegúrate de que las 18 fuentes estén cargadas en el notebook
2. Para cada podcast, **deselecciona** las fuentes que no correspondan al tema
3. Genera el audio overview (3-5 minutos de procesamiento por podcast)
4. Descarga y etiqueta claramente cada archivo de audio

---

