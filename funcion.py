import numpy as np
import math
import os
import cv2
import gradio as gr
from scipy.spatial.distance import euclidean
from PIL import Image
from procesamiento import reduce_brightness_contrast, gaussian_smoothing, kmeans_quantization, interseccion_lineas, ordenar_puntos, perspective_crop_with_filter, find_best_match
images_path = './Photos/'
dataset_dir = "./Dataset/"


def process_image(image):

    numpy_image = np.array(image)
    # Convertir la imagen de BGR a RGB para mostrarla correctamente en matplotlib
    image_rgb = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

    # Reducir el brillo y el contraste de la imagen
    adjusted_image = reduce_brightness_contrast(image_rgb, 100, 100)

    imagenquantized= kmeans_quantization(adjusted_image, n_colors=16, max_iter=10, epsilon=0.2)

    # Convertir a RGB y obtener la imagen en escala de grises
    gray = cv2.cvtColor(imagenquantized, cv2.COLOR_RGB2GRAY)

    # Suavizar la imagen en escala de grises
    gray2 = gaussian_smoothing(gray, 2, 5)

    # Aplicar el algoritmo de Canny con umbrales restrictivos
    edges = cv2.Canny(gray2, 50, 150, apertureSize=3)

    # Buscar líneas usando la transformada de Hough con umbral alto
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=125)

    #---------------------------------------------------------------------------------------------

    # Definir los rangos de ángulos para líneas horizontales y verticales
    vertical_min_angle = np.deg2rad(75)
    vertical_max_angle = np.deg2rad(105)
    horizontal_min_angle_1 = np.deg2rad(-10)
    horizontal_max_angle_1 = np.deg2rad(10)
    horizontal_min_angle_2 = np.deg2rad(170)
    horizontal_max_angle_2 = np.deg2rad(190)

    # Crear una copia de la imagen original
    copia4 = imagenquantized.copy()
    copia5 = imagenquantized.copy()

    # Listas para almacenar las líneas filtradas
    filtered_lines = []
    vertical_lines = []
    horizontal_lines = []

    if lines is not None:
        for i in range(len(lines)):
            # Obtener coordenadas polares
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            # Definir los umbrales de cercanía para líneas duplicadas
            rho_threshold = 4
            00     # Diferencia máxima permitida en rho
            theta_threshold = np.deg2rad(20)  # Diferencia máxima permitida en theta (5 grados)

            # Convertir de coordenadas polares a cartesianas
            a = math.cos(theta)
            b = math.sin(theta)

            x0 = a * rho
            y0 = b * rho

            # Obtener dos puntos en la línea para visualizarla
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            pt1 = (x1, y1)
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            pt2 = (x2, y2)

            # Verificar si la línea actual está dentro del rango permitido
            if (vertical_min_angle < theta < vertical_max_angle or
                horizontal_min_angle_1 < theta < horizontal_max_angle_1 or
                horizontal_min_angle_2 < theta < horizontal_max_angle_2):

                # Verificar si la línea actual es cercana a alguna en `filtered_lines`
                is_duplicate = False
                for (rho_f, theta_f) in filtered_lines:
                    # Comparar cercanía en términos de rho y theta
                    rho_diff = abs(rho - rho_f)
                    theta_diff = abs(theta - theta_f)
                    
                    # Normalizar theta_diff para considerar el ángulo en el rango de 0 a 2pi
                    if theta_diff > math.pi:
                        theta_diff = 2 * math.pi - theta_diff
                
                    # Si ambas diferencias están dentro de los umbrales, se considera un duplicado
                    if rho_diff < rho_threshold and theta_diff < theta_threshold:
                        is_duplicate = True
                        break
                
                # Si la línea no es duplicada, agregarla
                if not is_duplicate:
                    filtered_lines.append((rho, theta))
                    if vertical_min_angle < theta < vertical_max_angle:
                        vertical_lines.append((rho, theta))
                    elif horizontal_min_angle_1 < theta < horizontal_max_angle_1 or horizontal_min_angle_2 < theta < horizontal_max_angle_2:
                        horizontal_lines.append((rho, theta))
                                    # Convertir de coordenadas polares a cartesianas
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
            
                    # Obtener dos puntos en la línea para visualizarla
                    x1 = int(x0 + 2000 * (-b))
                    y1 = int(y0 + 2000 * (a))
                    pt1 = (x1, y1)
                    x2 = int(x0 - 2000 * (-b))
                    y2 = int(y0 - 2000 * (a))
                    pt2 = (x2, y2)
            
                    # Dibujar la línea en la imagen RGB
                    cv2.line(copia4, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)

    # Filtrar las dos líneas verticales más cercanas al borde izquierdo y derecho
    vertical_lines_sorted = sorted(vertical_lines, key=lambda x: abs(x[0]), reverse=True)
    horizontal_lines_sorted = sorted(horizontal_lines, key=lambda x: abs(x[0]), reverse=True)

    # Seleccionar las dos más cercanas
    selected_vertical_lines = []
    selected_horizontal_lines = []
    selected_vertical_lines.append(vertical_lines_sorted[0])
    selected_vertical_lines.append(vertical_lines_sorted[-1])
    selected_horizontal_lines.append(horizontal_lines_sorted[0])
    selected_horizontal_lines.append(horizontal_lines_sorted[-1])

    # Dibujar las líneas seleccionadas en copia2
    for rho, theta in selected_vertical_lines:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv2.line(copia5, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    for rho, theta in selected_horizontal_lines:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv2.line(copia5, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)

    # ----------------------------------------------------------------------------------------
    # Calcular los puntos de intersección entre cada recta vertical y horizontal
    puntos_interseccion = []

    for rho_v, theta_v in selected_vertical_lines:
        for rho_h, theta_h in selected_horizontal_lines:
            punto = interseccion_lineas(rho_v, theta_v, rho_h, theta_h)
            puntos_interseccion.append(punto)

    puntos_interseccion_ord = ordenar_puntos(puntos_interseccion)

    puntos_interseccion = []

    x1, y1 = puntos_interseccion_ord [0]
    x2, y2 = puntos_interseccion_ord [1]
    x3, y3 = puntos_interseccion_ord [2]
    x4, y4 = puntos_interseccion_ord [3]

    # Llamar a la función
    imagen_cortada = perspective_crop_with_filter(image_rgb,x1, y1, x2, y2, x3, y3, x4, y4, w_window=3, apply_filter=True)

    best_match, score = find_best_match(dataset_dir, imagen_cortada)
    print(f"Best match: {best_match} with score: {-1*score}")

    best_match_path = os.path.join(dataset_dir, best_match)
    image_name = os.path.basename(best_match_path)
    dataset_image = cv2.imread(best_match_path)
    
    
    
    dataset_image_rgb = cv2.cvtColor(dataset_image, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(dataset_image_rgb)
    
    
#----------------------------------------------------------------------------------------   
    
    info_dir = "./Info/"
    

    # Nombre de la imagen, como parámetro (por ejemplo, 'image_name = 'Breakfast in America-Supertramp'')
    # Añadir extensión .jpeg si no tiene extensión
    image_path = os.path.join(info_dir, image_name[:-4])

    # Verificar si la imagen tiene extensión, si no, añadir .jpeg como predeterminado
    if not image_path.lower().endswith('.jpeg'):
        image_path += '.jpeg'

    # Verificar si el archivo existe
    if not os.path.exists(image_path):
        print(f"El archivo {image_path} no existe.")
    else:
        # Cargar la imagen
        info_image = cv2.imread(image_path)
        
        if info_image is None:
            print("Error al cargar la imagen.")
        else:
            # Convertir de BGR a RGB
            info_image_rgb = cv2.cvtColor(info_image, cv2.COLOR_BGR2RGB)

            # Convertir a imagen de Pillow
            pil_info_image = Image.fromarray(info_image_rgb)

            # Asumiendo que tienes otra imagen para procesar (por ejemplo, dataset_image_rgb)
            dataset_image_rgb = info_image_rgb  # O la imagen que estés usando
            pil_image_info = Image.fromarray(dataset_image_rgb)

#----------------------------------------------------------------------------------------
    
    return pil_image, pil_image_info, image_name[:-4]