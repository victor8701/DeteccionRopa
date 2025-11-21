import cv2
from ultralytics import YOLO
import sys
import os

def detectar_ropa():
    print("--- Cargando YOLO-World (Modelo de Vocabulario Abierto) ---")
    
    # Cargamos el modelo
    try:
        model = YOLO('yolov8s-world.pt')  
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        sys.exit(1)

    # Definimos qué buscar
    mis_clases = [
        "bag", "belt", "boots", "footwear", 
        "outer clothing", "dress", "sunglasses", 
        "pants", "top", "shorts", "skirt", 
        "headwear", "scarf", "tie"
    ]
    
    print(f"Configurando clases: {mis_clases}")
    model.set_classes(mis_clases)

    # RUTA BASE
    carpeta_base = "/home/ubuntu20/Ubuntu20_ws/src/DeteccionRopa/images"
    #carpeta_base = "/home/ubuntu20/Ubuntu20_ws/src/DeteccionRopa/src/Clothing_Detection_YOLO/tests"

    print(f"\nBuscando en: {carpeta_base}")
    print("Introduce el nombre de la imagen (sin .jpg):")
    
    try:
        nombre = input().strip()
    except EOFError:
        return

    ruta_imagen = os.path.join(carpeta_base, f"{nombre}.jpg")
    
    if not os.path.exists(ruta_imagen):
        print(f"ERROR: No existe {ruta_imagen}")
        return

    print(f"Procesando imagen...")

    # PREDICCIÓN
    # save=True guarda la imagen en disco también
    results = model.predict(source=ruta_imagen, save=True, conf=0.15)

    # -----------------------------------------------------------
    # PARTE NUEVA: MOSTRAR LA IMAGEN EN PANTALLA
    # -----------------------------------------------------------
    
    # Generamos la imagen con los dibujos (cajas y probabilidades)
    imagen_resultados = results[0].plot()

    print("\n--- OBJETOS DETECTADOS ---")
    for r in results:
        for c in r.boxes.cls:
            nombre = model.names[int(c)]
            conf = r.boxes.conf[0]
            print(f"- {nombre}: {conf:.2f} ({int(conf*100)}%)")

    print("\nIntentando abrir ventana gráfica...")
    try:
        # Título de la ventana
        nombre_ventana = "Resultado Deteccion Ropa"
        
        # Permitir redimensionar la ventana (útil si la foto es enorme)
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
        
        # Mostrar la imagen
        cv2.imshow(nombre_ventana, imagen_resultados)
        
        print("✅ Ventana abierta. Pulsa cualquier tecla sobre la imagen para cerrar.")
        # Esperar tecla
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        # Si falla (común en WSL sin entorno gráfico configurado)
        print("\n⚠️ AVISO: No se pudo abrir la ventana en este terminal.")
        print(f"ERROR TÉCNICO: {e}")
        print("Pero NO PASA NADA. La imagen con los recuadros se ha guardado aquí:")
        print(f"👉 {results[0].save_dir}")
        print("Puedes ir a esa carpeta y abrir el archivo .jpg resultante.")

if __name__ == "__main__":
    detectar_ropa()