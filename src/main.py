#!/usr/bin/env python3
"""
Sistema Integrado de Detección de Ropa y Esquinas de Caja

Este script coordina dos módulos:
1. detectarRopaYolov8: Detecta prendas de ropa y sus centros
2. DeteccionRopa: Detecta esquinas de la caja y dibuja los centros de las prendas

Autor: Sistema de detección para robótica
Fecha: 2025-12-06
"""

import os
import sys

# ============================================================================
# CONFIGURACIÓN CENTRALIZADA - MODIFICAR AQUÍ PARA AJUSTAR EL SISTEMA
# ============================================================================

# --- VISUALIZACIÓN ---
MOSTRAR_VENTANA_YOLO = True  # True: Muestra ventana con detecciones YOLO
                             # False: No muestra ventana (más rápido)

MOSTRAR_VENTANA_ESQUINAS = True  # True: Muestra ventana con esquinas detectadas
                                  # False: No muestra ventana

# --- GUARDADO DE ARCHIVOS ---
GUARDAR_IMAGENES_YOLO = False  # True: Guarda imágenes en runs/detect/predictX/
                              # False: No guarda imágenes

GUARDAR_REPORTE_COMPLETO = False  # True: Guarda reporte unificado en resultados_deteccion/
                                 # False: No guarda reporte

# --- DETECCIÓN DE PRENDAS (YOLO) ---
CONFIANZA_MINIMA = 0.5  # Umbral de confianza mínima (0.0 - 1.0)
                        # Valores más altos = menos detecciones pero más precisas
                        # Valores más bajos = más detecciones pero con más errores
                        # Recomendado: 0.5 - 0.75

IOU_THRESHOLD = 0.3  # Umbral de IoU para eliminar duplicados (0.0 - 1.0)
                     # Valores más bajos = más estricto (elimina más duplicados)
                     # Valores más altos = más permisivo (permite más solapamiento)
                     # Recomendado: 0.3

# --- DETECCIÓN DE ESQUINAS ---
MARGEN_EXCLUSION_PRENDAS = 20  # Margen en píxeles alrededor de cada prenda
                               # para excluir de la detección de esquinas
                               # Valores más altos = excluye más área (menos esquinas falsas)
                               # Valores más bajos = excluye menos área (más esquinas)
                               # Recomendado: 15 - 30

MAX_ESQUINAS = 20  # Número máximo de esquinas a detectar
                   # Aumentar si necesitas detectar más esquinas
                   # Recomendado: 15 - 25

QUALITY_LEVEL_ESQUINAS = 0.05  # Calidad mínima de las esquinas (0.0 - 1.0)
                               # Valores más altos = solo esquinas muy definidas
                               # Valores más bajos = detecta esquinas más débiles
                               # Recomendado: 0.01 - 0.1

MIN_DISTANCE_ESQUINAS = 70  # Distancia mínima entre esquinas (píxeles)
                            # Aumentar para evitar esquinas muy juntas
                            # Recomendado: 50 - 100

# --- RUTAS ---
CARPETA_IMAGENES = "02Dic"  # Subcarpeta dentro de images/ donde están las imágenes
                            # Cambiar según tu estructura de carpetas

# ============================================================================
# FIN DE CONFIGURACIÓN
# ============================================================================

# Importar los módulos de detección
from deteccion_ropa import detectar_ropa
from deteccion_esquinas import charge_image

def sistema_completo(nombre_imagen=None):
    """
    Ejecuta el sistema completo de detección.
    
    Proceso:
    1. Detecta prendas con YOLO y obtiene sus centros
    2. Detecta esquinas de la caja y dibuja los centros de las prendas
    
    Args:
        nombre_imagen: Nombre de la imagen (sin extensión) o None para pedir al usuario
    """
    print("="*60)
    print("SISTEMA INTEGRADO DE DETECCIÓN")
    print("="*60)
    
    # Construir ruta de imagen
    if nombre_imagen is None:
        print("\nIntroduce el nombre de la imagen (sin .jpg):")
        nombre_imagen = input().strip()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    carpeta_imagenes = os.path.join(script_dir, "..", "images", CARPETA_IMAGENES)
    carpeta_imagenes = os.path.abspath(carpeta_imagenes)
    ruta_imagen = os.path.join(carpeta_imagenes, f"{nombre_imagen}.jpg")
    
    if not os.path.exists(ruta_imagen):
        print(f"❌ ERROR: No existe la imagen {ruta_imagen}")
        return
    
    print(f"\n📷 Procesando imagen: {nombre_imagen}.jpg")
    print(f"   Ruta: {ruta_imagen}")
    
    # =========================================================================
    # PASO 1: Detectar prendas con YOLO
    # =========================================================================
    print("\n" + "="*60)
    print("PASO 1: DETECCIÓN DE PRENDAS CON YOLO")
    print("="*60)
    
    prendas_detectadas = detectar_ropa(
        ruta_imagen=ruta_imagen,
        mostrar_ventana=MOSTRAR_VENTANA_YOLO,
        guardar_archivos=GUARDAR_IMAGENES_YOLO,
        confianza_minima=CONFIANZA_MINIMA,
        iou_threshold=IOU_THRESHOLD
    )
    
    if not prendas_detectadas or len(prendas_detectadas) == 0:
        print("\n⚠️  No se detectaron prendas. Continuando con detección de esquinas...")
    else:
        print(f"\n✅ Detectadas {len(prendas_detectadas)} prendas:")
        for item in prendas_detectadas:
            # Desempaquetar según formato (4 o 8 elementos)
            if len(item) == 8:
                nombre, conf, cx, cy, x1, y1, x2, y2 = item
            else:
                nombre, conf, cx, cy = item
            print(f"   - {nombre}: {conf:.2f} ({int(conf*100)}%) en ({cx}, {cy})")
    
    # =========================================================================
    # PASO 2: Detectar esquinas y dibujar centros de prendas
    # =========================================================================
    print("\n" + "="*60)
    print("PASO 2: DETECCIÓN DE ESQUINAS Y VISUALIZACIÓN")
    print("="*60)
    
    print("\n🔍 Detectando esquinas de la caja...")
    print("🎯 Dibujando centros de prendas en VERDE...")
    
    imagen_resultado = charge_image(
        ruta_imagen=ruta_imagen,
        prendas_detectadas=prendas_detectadas,
        mostrar_ventana=MOSTRAR_VENTANA_ESQUINAS,
        guardar_reporte=GUARDAR_REPORTE_COMPLETO,
        margen_exclusion=MARGEN_EXCLUSION_PRENDAS,
        max_esquinas=MAX_ESQUINAS,
        quality_level=QUALITY_LEVEL_ESQUINAS,
        min_distance=MIN_DISTANCE_ESQUINAS
    )
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)
    print(f"📊 Prendas detectadas: {len(prendas_detectadas) if prendas_detectadas else 0}")
    print(f"📁 Reporte guardado en: runs/detect/predictX/detecciones.txt")
    print(f"🖼️  Imagen YOLO guardada en: runs/detect/predictX/{nombre_imagen}.jpg")
    print(f"👁️  Visualización: Imagen con esquinas (rojos) y centros de prendas (verdes)")
    print("="*60)

if __name__ == "__main__":
    # Si se pasa un argumento, usarlo como nombre de imagen
    if len(sys.argv) > 1:
        nombre = sys.argv[1]
        sistema_completo(nombre)
    else:
        # Modo interactivo
        sistema_completo()
