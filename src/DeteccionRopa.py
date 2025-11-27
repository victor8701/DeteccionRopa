import cv2
import numpy as np
import sys
import os

BASE_PATH = "../images/"

# Variables globales para mantener la estructura del C++
# En Python no son estrictamente necesarias si pasamos argumentos, 
# pero las dejo para imitar tu código original.
imagen = None       # Color
imageGray = None    # Gris
imageEdited = None  # Resultado

def border_sobel(gray_img):
    """
    Aplica el filtro Sobel en X e Y y los combina.
    Retorna la imagen con los bordes detectados.
    """
    # Gradiente en X
    # CV_16S para evitar desbordamiento con valores negativos
    grad_x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    
    # Gradiente en Y
    grad_y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Combinar ambos gradientes
    combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return combined

def obtener_mascara_carton(imagen_bgr):
    """
    Crea una máscara estricta para el color cartón,
    eliminando grises (metal) y oscuros (fondo).
    """
    # Convertir a HSV
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    
    # --- AJUSTE RESTRICTIVO ---
    # Hue (10-30): Rango naranja/marrón específico.
    # Sat (65-255): ALTA saturación mínima para evitar grises/metales.
    # Val (60-255): Brillo medio-alto para evitar sombras oscuras.
    lower_brown = np.array([10, 65, 60]) 
    upper_brown = np.array([30, 255, 255])
    
    # Crear la máscara (255 si está en el rango, 0 si no)
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # --- LIMPIEZA DE RUIDO ---
    kernel = np.ones((5, 5), np.uint8)
    
    # 1. Eliminar ruido blanco fuera de la caja (puntitos aislados)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2. Cerrar huecos dentro de la caja
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 3. EROSIÓN (Importante): Encoge la máscara ligeramente.
    # Esto asegura que el borde detectado sea el del cartón y no el borde
    # que toca el metal. Evita que la máscara "sangre" hacia el riel.
    mask = cv2.erode(mask, kernel, iterations=2)
    
    return mask

def detectar_esquinas_caja(sobel_img, original_img, mask_color):
    """
    Detecta esquinas y muestra solo la caja con puntos grandes.
    """
    height, width = sobel_img.shape[:2]
    
    # 1. Parámetros de detección
    max_corners = 20
    quality_level = 0.1
    min_distance = 70
    
    # 2. Detección sobre el resultado de Sobel
    # corners devuelve un array numpy de forma (N, 1, 2)
    corners = cv2.goodFeaturesToTrack(
        sobel_img, 
        maxCorners=max_corners, 
        qualityLevel=quality_level, 
        minDistance=min_distance,
        blockSize=3, 
        useHarrisDetector=False, 
        k=0.04,
        mask=mask_color  # Solo busca donde la máscara es blanca
    )

    # CREAR VISUALIZACIÓN: FONDO NEGRO
    # Usamos bitwise_and para que solo se vean los píxeles de la caja
    output_img = cv2.bitwise_and(original_img, original_img, mask=mask_color)

    if corners is not None:
        # Convertimos a lista para poder ordenar fácilmente
        corners_list = list(corners)

        # 3. Ordenar las esquinas respecto al origen (0,0) INFERIOR IZQUIERDA
        # Lógica C++: dist = x^2 + (altura - y)^2
        # c[0][0] es X, c[0][1] es Y
        corners_list.sort(key=lambda c: (c[0][0]**2) + (height - c[0][1])**2)

        print("--- ESQUINAS ORDENADAS (Filtradas por color cartón) ---")
        
        for i, corner in enumerate(corners_list):
            x, y = corner.ravel() # aplanar array a (x, y)
            
            # Coordenadas transformadas para mostrar por pantalla
            x_real = x
            y_real = height - y
            
            print(f"Punto {i + 1}: ({x_real:.2f}, {y_real:.2f})")
            
            # Dibujar círculo rojo GRANDE (Radio 12)
            center = (int(x), int(y))
            
            # Radio aumentado a 12 (antes era 5)
            cv2.circle(output_img, center, 12, (0, 0, 255), -1)
            
            # Texto un poco más grueso y desplazado
            cv2.putText(output_img, str(i + 1), (int(x) + 15, int(y) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        print("-" * 52)
        return output_img
    else:
        print("No se encontraron esquinas en el área del color especificado.")
        return output_img

def charge_image():
    global imagen, imageGray, imageEdited

    print("Enter image name (.jpg format)")
    image_name = input() # Espera entrada del usuario
    
    # Construcción de la ruta
    # Si la carpeta es local, puedes quitar BASE_PATH y poner 'images/' + ...
    path = os.path.join(BASE_PATH, image_name + ".jpg")
    
    # Intentar cargar imagen (la ruta debe ser correcta o fallará)
    # Comprobamos si existe el archivo primero para evitar error de opencv
    if not os.path.isfile(path):
        # Fallback por si acaso la ruta absoluta no funciona, probamos local
        path = f"images/{image_name}.jpg"
        if not os.path.isfile(path):
            print(f"Error al cargar la imagen: {path}")
            sys.exit(1)

    # Cargar en escala de grises
    imageGray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Cargar en color
    imagen = cv2.imread(path, cv2.IMREAD_COLOR)

    if imagen is None:
        print(f"Error al cargar la imagen (formato incorrecto o vacía): {path}")
        sys.exit(1)

    # 1. Aplicamos el filtro Sobel
    # Esto llena 'imageEdited' con los bordes detectados
    # En Python retornamos la matriz en vez de escribir en global directamente
    imagen_sobel = border_sobel(imageGray)

    # 2. Detectamos esquinas y ordenamos
    # Pasamos la imagen sobel para detectar y la original para pintar
    mascara_carton = obtener_mascara_carton(imagen)

    # 3. Detectamos esquinas y generamos la imagen procesada (Caja sobre fondo negro)
    image_result = detectar_esquinas_caja(imagen_sobel, imagen, mascara_carton)

    # 4. Concatenamos:
    # IZQUIERDA: Imagen Original (imagen)
    # DERECHA: Imagen Procesada (image_result)
    imagen_combinada = cv2.hconcat([imagen, image_result])

    cv2.namedWindow("Izq: Original | Der: Procesada (Solo Caja)", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Izq: Original | Der: Procesada (Solo Caja)", imagen_combinada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    charge_image()