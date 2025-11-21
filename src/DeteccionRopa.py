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

def detectar_esquinas_caja(sobel_img, original_img):
    """
    Detecta esquinas sobre la imagen Sobel, las ordena desde la 
    esquina inferior-izquierda y dibuja sobre la imagen original.
    """
    height, width = sobel_img.shape[:2]
    
    # 1. Parámetros de detección
    max_corners = 20
    quality_level = 0.1
    min_distance = 50
    
    # 2. Detección sobre el resultado de Sobel
    # corners devuelve un array numpy de forma (N, 1, 2)
    corners = cv2.goodFeaturesToTrack(
        sobel_img, 
        maxCorners=max_corners, 
        qualityLevel=quality_level, 
        minDistance=min_distance,
        blockSize=3, 
        useHarrisDetector=False, 
        k=0.04
    )

    if corners is not None:
        # Convertimos a lista para poder ordenar fácilmente
        corners_list = list(corners)

        # 3. Ordenar las esquinas respecto al origen (0,0) INFERIOR IZQUIERDA
        # Lógica C++: dist = x^2 + (altura - y)^2
        # c[0][0] es X, c[0][1] es Y
        corners_list.sort(key=lambda c: (c[0][0]**2) + (height - c[0][1])**2)

        # 4. Visualización
        # Copiamos la imagen original para pintar resultados
        output_img = original_img.copy()

        print("--- ESQUINAS ORDENADAS (Desde Inferior-Izquierda) ---")
        
        for i, corner in enumerate(corners_list):
            x, y = corner.ravel() # aplanar array a (x, y)
            
            # Coordenadas transformadas para mostrar por pantalla
            x_real = x
            y_real = height - y
            
            print(f"Punto {i + 1}: ({x_real:.2f}, {y_real:.2f})")
            
            # Dibujar círculo rojo
            center = (int(x), int(y))
            cv2.circle(output_img, center, 5, (0, 0, 255), -1)
            
            # Escribir número
            cv2.putText(output_img, str(i + 1), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        print("-" * 52)
        return output_img
    else:
        print("No se encontraron esquinas.")
        return original_img.copy()

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
    image_result = detectar_esquinas_caja(imagen_sobel, imagen)

    # Convertimos Sobel a BGR para poder concatenar con la imagen a color
    imagen_sobel_color = cv2.cvtColor(imagen_sobel, cv2.COLOR_GRAY2BGR)

    # Concatenamos horizontalmente (hconcat)
    imagen_combinada = cv2.hconcat([imagen_sobel_color, image_result])

    # Mostrar ventana
    cv2.namedWindow("Panel de Control", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Panel de Control", imagen_combinada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    charge_image()