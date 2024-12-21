# Vynilizer 🎵💿

**Vynilizer** es una aplicación web desarrollada en Python que permite a los usuarios identificar discos de vinilo mediante una fotografía. Ya sea desde el móvil o el ordenador, puedes subir una imagen de un vinilo, y la app analizará la foto para ofrecerte información sobre el disco.

## 🚀 Descripción

El objetivo de Vynilizer es brindar una herramienta sencilla y rápida para los amantes del vinilo, ayudándolos a reconocer discos de su colección o desconocidos.  

Mediante técnicas de visión por computadora como detección de bordes, detección de líneas con Hough, y SIFT para el reconocimiento de características, Vynilizer analiza la imagen y la compara con una base de datos de discos.  

Actualmente, solo se reconocen los discos incluidos en su base de datos limitada.

## 🌟 Características

- Subir o capturar una imagen de un vinilo.
- Reconocimiento de líneas y patrones en los discos mediante Hough y SIFT.
- Comparación con una base de datos para identificar el disco.
- Interfaz intuitiva desarrollada con **Gradio** para facilitar el uso.

## 🛠️ Tecnologías Utilizadas

- **Lenguaje:** Python
- **Interfaz de Usuario:** Gradio
- **Visión Computacional:** OpenCV (cv2), NumPy
- **Reconocimiento de Características:** SIFT
- **Detección de Líneas:** Transformada de Hough

## 📝 Instalación

1. Clona este repositorio:
   ```bash
   git clone (repositorio)

   cd vynilizer app

   instala Python, Gradio, OpenCV, Numpy, os, math

   python app.py

   Accede a la interfaz en tu navegador en la dirección dada
   
##  📖 Uso
   Captura o sube una imagen de un vinilo a través de la interfaz.
   La aplicación procesará la imagen:
   Detectará bordes y líneas del vinilo.
   Utilizará SIFT para comparar características del disco con la base de datos.
   Si el vinilo está en la base de datos, se mostrará información como título, artista y año.

##   📅 Roadmap
   Ampliar la base de datos de discos disponibles.
   Optimizar el reconocimiento para mejorar la precisión.
   Añadir soporte para cargar información de vinilos de forma colaborativa.
   Mejorar la interfaz para adaptarse a dispositivos móviles.

##   🤝 Contribuciones
   ¡Las contribuciones son bienvenidas!
   Si tienes ideas para mejorar el proyecto o deseas colaborar, no dudes en abrir un issue o enviar un pull request.