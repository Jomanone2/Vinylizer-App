import numpy as np
import math
import os
import cv2
from scipy.spatial.distance import euclidean

images_path = './Photos/'
dataset_dir = "./Dataset/"

def reduce_brightness_contrast(image, brightness, contrast):
    """
    Reduce el brillo y el contraste de una imagen.

    :param image: Imagen de entrada en formato numpy.
    :param brightness: Valor de brillo a reducir (entre 0 y 100).
    :param contrast: Valor de contraste a reducir (entre 0 y 100).
    :return: Imagen con brillo y contraste reducidos.
    """
    # Calcular el factor de contraste (escala de 0 a 1 para suavizar el cambio)
    alpha = 1.0 + (contrast / 100.0)
    # Calcular el factor de brillo (positivo o negativo según el brillo deseado)
    beta = -brightness

    # Aplicar el cambio de brillo y contraste a la imagen
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return adjusted_image

def gaussian_smoothing(image, sigma, w_kernel):
    """ Blur and normalize input image.   
    
        Args:
            image: Input image to be binarized
            sigma: Standard deviation of the Gaussian distribution
            w_kernel: Kernel aperture size
                    
        Returns: 
            binarized: Blurred image
    """   
    
    # Define 1D kernel
    s=sigma
    w=w_kernel
    kernel_1D = [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-w,w+1)]
    
    # Apply distributive property of convolution
    kernel_2D = np.outer(kernel_1D,kernel_1D)
    
    # Blur image
    smoothed_img = cv2.filter2D(image,cv2.CV_8U,kernel_2D)
    
    # Normalize to [0 254] values
    smoothed_norm = np.array(image.shape)
    smoothed_norm = cv2.normalize(smoothed_img,None, 0, 255, cv2.NORM_MINMAX)
    
    return smoothed_norm

def polar_to_cartesian(r, theta):
    """
    Convierte coordenadas polares a cartesianas.

    Args:
        r (float): La distancia radial desde el origen.
        theta (float): El ángulo en radianes desde el eje positivo X.

    Returns:
        tuple: Coordenadas cartesianas (x, y).
    """
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y
    
def dibujar_linea(copia, x1, y1, x2, y2, color, grosor=3):
    # Calcular la pendiente de la línea
    dy = y2 - y1
    dx = x2 - x1

    # Evitar división por cero
    if dx == 0:
        # Línea completamente vertical
        cv2.line(copia, (x1, 0), (x1, copia.shape[0]), color, grosor, cv2.LINE_AA)
    else:
        # Línea inclinada
        m = dy / dx  # pendiente
        b = y1 - m * x1  # intercepto en el eje y (y = mx + b)

        # Calcular extremos de la línea en los bordes de la imagen
        x_start, x_end = 0, copia.shape[1]  # Borde izquierdo y derecho de la imagen
        y_start = int(m * x_start + b)
        y_end = int(m * x_end + b)

        # Dibujar la línea extendida
        cv2.line(copia, (x_start, y_start), (x_end, y_end), color, grosor, cv2.LINE_AA)

def interseccion_lineas(rho1, theta1, rho2, theta2):
    # Convertir las ecuaciones polares a cartesianas
    A = np.array([[math.cos(theta1), math.sin(theta1)],
                  [math.cos(theta2), math.sin(theta2)]])
    b = np.array([rho1, rho2])
    
    # Resolver el sistema de ecuaciones
    interseccion = np.linalg.solve(A, b)
    return interseccion

def perspective_crop_with_filter(image_rgb,x1, y1, x2, y2, x3, y3, x4, y4, w_window=3, apply_filter=True):
    # Definir los puntos de la imagen original basados en los valores de las barras o entradas
    pts_original = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # Definir los puntos del destino final (un rectángulo recto)
    pts_destino = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    # Obtener la matriz de transformación de perspectiva
    M = cv2.getPerspectiveTransform(pts_original, pts_destino)

    # Aplicar la transformación a la imagen original
    warped_image = cv2.warpPerspective(image_rgb, M, (300, 300))
    
    # Si se selecciona, aplicar el filtro de mediana
    #if apply_filter:
    #    warped_image = median_filter(warped_image, w_window)
    
    # Redimensionar la imagen
    warped_scaled_image = cv2.resize(warped_image, (0, 0), fx=2, fy=2)  # Escalar al doble

    # Mostrar la imagen original con los puntos
    #plt.figure(figsize=(10, 5))
    
    #Mostrar la imagen original
    #plt.subplot(1, 2, 1)
    #plt.imshow(image_rgb)
    #plt.title("Imagen con puntos")
    
    # Dibujar los puntos seleccionados
    #for point in pts_original:
    #    plt.plot(point[0], point[1], 'ro')  # Puntos rojos en las esquinas
    
    # Mostrar la imagen transformada (con o sin filtro)
    #plt.subplot(1, 2, 2)
    #plt.imshow(warped_image)
    #plt.title("Imagen transformada (con filtro de mediana)" if apply_filter else "Imagen transformada")

    #plt.show()

    #print("Matriz de transformación")
    #print(M)

    return warped_image
    
def ordenar_puntos(puntos):
    puntos = np.array(puntos, dtype="float32")
    
    # Inicializar el array de puntos ordenados
    puntos_ordenados = np.zeros((4, 2), dtype="float32")
    
    # Calcular la suma y la diferencia de las coordenadas
    s = puntos.sum(axis=1)  # Suma de las coordenadas (x + y)
    diff = np.diff(puntos, axis=1)  # Diferencia entre x e y (x - y)
    
    # Esquina superior izquierda (menor suma), inferior derecha (mayor suma)
    puntos_ordenados[0] = puntos[np.argmin(s)]  # Esquina superior izquierda
    puntos_ordenados[3] = puntos[np.argmax(s)]  # Esquina inferior derecha
    
    # Esquinas superior derecha e inferior izquierda
    puntos_ordenados[1] = puntos[np.argmin(diff)]  # Esquina superior derecha
    puntos_ordenados[2] = puntos[np.argmax(diff)]  # Esquina inferior izquierda
    
    return puntos_ordenados

def extract_features(image):
    features = {}

    # SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    features['sift'] = descriptors

    # Harris Corner Detector
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    features['harris'] = harris

    # Histogram Moments
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    features['hist_moments'] = cv2.moments(hist.flatten())

    return features

def compare_features(features1, features2):
    scores = {}

    # SIFT
    if features1['sift'] is not None and features2['sift'] is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(features1['sift'], features2['sift'], k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        scores['sift'] = len(good_matches)
    else:
        scores['sift'] = 0

    # Harris (Euclidean distance between corner responses)
    scores['harris'] = euclidean(features1['harris'].flatten(), features2['harris'].flatten())

    # Histogram Moments (Euclidean distance)
    scores['hist_moments'] = euclidean(
        np.array(list(features1['hist_moments'].values())),
        np.array(list(features2['hist_moments'].values()))
    )

    return scores

def find_best_match(dataset_dir, input_image):
    input_features = extract_features(input_image)

    best_match = None
    best_score = float('inf')

    for image_name in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image_name)
        #print(f"Processing: {image_path}")  # Debugging line
        dataset_image = cv2.imread(image_path)

        if dataset_image is None:
            #print(f"Failed to load image: {image_name}")
            continue

        dataset_features = extract_features(dataset_image)

        scores = compare_features(input_features, dataset_features)

        # Combine scores: lower is better
        total_score = - scores['sift']
        
        if total_score < best_score:
            best_score = total_score
            best_match = image_name

    return best_match, best_score

def kmeans_quantization(image, n_colors=16, max_iter=30, epsilon=0.2):
    """
    Aplica la cuantización de colores a una imagen usando K-means.

    Parámetros:
    - image: np.ndarray -> Imagen de entrada.
    - n_colors: int -> Número de colores (clusters) deseados en la salida.
    - max_iter: int -> Número máximo de iteraciones para el algoritmo K-means.
    - epsilon: float -> Precisión deseada para la convergencia.

    Retorna:
    - quantized_image: np.ndarray -> Imagen cuantizada con n_colors colores.
    """
    # Aplanar la imagen a un array 2D y convertir a tipo float32
    flattened_img = image.reshape((-1, 3))
    flattened_img = np.float32(flattened_img)

    # Configurar criterios para K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

    # Aplicar K-means clustering
    _, labels, centers = cv2.kmeans(flattened_img, n_colors, None, criteria, max_iter, cv2.KMEANS_RANDOM_CENTERS)

    # Convertir centros a tipo uint8
    centers = np.uint8(centers)

    # Asignar a cada píxel su color (centro de su cluster)
    res = centers[labels.flatten()]

    # Reconstruir la imagen cuantizada
    quantized_image = res.reshape((image.shape))

    return quantized_image

