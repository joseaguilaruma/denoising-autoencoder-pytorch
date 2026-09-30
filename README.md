# 🔍 Denoising Autoencoder para Fashion-MNIST con PyTorch

Implementación de un **Autocodificador con eliminación de ruido (Denoising Autoencoder)** desarrollado en PyTorch. Este modelo es capaz de reconstruir imágenes de prendas de ropa a partir de entradas corrompidas con ruido artificial.

Proyecto correspondiente a la asignatura de Programación para la Inteligencia Artificial del Grado en Ingeniería Informática (Universidad de Málaga).

## 🎯 Objetivo del Proyecto
A diferencia de los enfoques supervisados tradicionales, este modelo de aprendizaje profundo aprende la función identidad sin necesidad de etiquetado clásico, enfocándose en la limpieza de ruido. 
Al conjunto de datos Fashion-MNIST original se le inyectó un ruido gaussiano de media 0 y desviación típica 0.2. El autocodificador se entrenó para filtrar este ruido y devolver las imágenes nítidas originales.

## 🛠️ Tecnologías y Librerías Utilizadas
*   **Lenguaje:** Python 3
*   **Deep Learning:** PyTorch, Torchvision
*   **Optimizador y Pérdida:** Adam (Learning Rate: 1e-3), MSELoss
*   **Visualización y Evaluación:** Matplotlib, Scikit-Learn (t-SNE)

## ⚙️ Arquitectura de la Red
El modelo consta de neuronas lineales y se divide en dos fases:
1.  **Encoder:** Comprime la entrada desde 784 píxeles pasando por capas ocultas de 512, 128 y 64 neuronas (con activaciones ReLU) hasta un **espacio latente de 8 dimensiones**.
2.  **Decoder:** Reconstruye la imagen pasando del espacio latente (8) hacia capas de 64, 128 y 512 neuronas, finalizando en los 784 píxeles originales mediante una activación Sigmoid.

## 🚀 Instalación y Ejecución
Para ejecutar este cuaderno en tu entorno local o en Google Colab:
1. Clona el repositorio: `git clone https://github.com/TU_USUARIO/denoising-autoencoder-pytorch.git`
2. Instala los requerimientos básicos: `pip install torch torchvision scikit-learn matplotlib numpy tqdm`
3. Ejecuta las celdas del archivo principal. El dataset de evaluación descargará las 10.000 imágenes de prueba automáticamente.

## 📈 Resultados Destacados
El modelo se entrenó mediante procesamiento en GPU (CUDA) con lotes de 1024 imágenes durante 200 épocas. A continuación se muestran los resultados visuales de la reconstrucción y el agrupamiento no supervisado en el espacio latente:

*(Añade aquí tus imágenes)*
![Evolución del Error](suppor-images/error-image.png)
![Espacio Latente (0 vs 1)](support-images/latent-space)
![Espacio Latente (2 vs 3)](support-images/latent-space-2)
