import gradio as gr
from PIL import Image
from funcion import process_image

def js_to_prefere_the_back_camera_of_mobilephones():
    custom_html = """
    <script>
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    
    navigator.mediaDevices.getUserMedia = (constraints) => {
      if (!constraints.video.facingMode) {
        constraints.video.facingMode = {ideal: "environment"};
      }
      return originalGetUserMedia(constraints);
    };
    </script>
    """
    return custom_html

# Función para procesar imágenes (ejemplo: aumentar el contraste)
def process_image_2(image):
    rotated_image = image.rotate(180, expand=True)
    return rotated_image

# Función para resetear los componentes
def reset_fields():
    return None, None, None, None  # Devuelve valores vacíos para los componentes de salida

# CSS personalizado para fondo azul oscuro y amarillo
custom_css = """
.gradio-container {
    background-color: #080501; /* Fondo azul oscuro */
    color: #FFD700; /* Amarillo brillante */
    font-family: 'Courier New', monospace;
    text-align: center;
}
.gr-button {
    background-color: #FFD700; /* Botón amarillo */
    color: #000000; /* Texto negro */
    border: 2px solid #FFD700;
    font-weight: bold;
    font-size: 16px;
    border-radius: 8px;
    transition: transform 0.2s;
}
.gr-button:hover {
    transform: scale(1.1); /* Botón crece al pasar el cursor */
    background-color: #FFC300; /* Amarillo más claro */
}
#upload {
    border: 2px dashed #FFD700; /* Bordes amarillos en la zona de subida */
    border-radius: 10px;
    padding: 10px;
}
#output {
    border: 2px solid #FFD700; /* Bordes amarillos en la imagen procesada */
    border-radius: 10px;
    margin-top: 20px;
}
"""

# Crear la interfaz
with gr.Blocks(css=custom_css, head=js_to_prefere_the_back_camera_of_mobilephones()) as demo:
    gr.Markdown(
        """
        # 🎵💿 **Vinylizer Project** 💿🎵  
        Hazle una foto a tu vinilo. Súbela, toca el botón, y ¡magia!  
        **Revive la información musical en la palma de tu mano.**
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil", 
                label="📸 Sube tu imagen de vinilo aquí", 
                elem_id="upload"
            )
        with gr.Column():
            output_image = gr.Image(
                label="✨ Imagen Vinylizada", 
                elem_id="output_image", 
                interactive=False
            )
            
            output_name = gr.Textbox( 
                label="📃 Nombre del album",
                interactive=False,  # El texto no es editable
                elem_id="output_name"
            )
            
            info_image = gr.Image(
                label="📝 Info", 
                elem_id="extra_image", 
                interactive=False
            )

    process_button = gr.Button("🚀 Procesar Vinilo")
    reset_button = gr.Button("🔄 Resetear")

    # Conectar entrada, botón y salida
    process_button.click(process_image, inputs=[input_image], outputs=[output_image, info_image, output_name])
    reset_button.click(reset_fields, inputs=[], outputs=[input_image, output_image, output_name, info_image])

# Lanzar la app
demo.launch(share=True)



