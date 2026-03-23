# Workshop 2 - IA (Machine Learning & Deep Learning)

Este repositorio contiene la solucion de los dos problemas

1. **Clasificación**: detección de fatiga muscular usando señales EMG (extracción de características + modelos de ML y una red neuronal).
2. **Regresión**: predicción de edad a partir de imágenes faciales usando una CNN

---

## Estructura del repositorio

```text
Workshop_2_IA/
├─ README.md
├─ clasificacion/
│  └─ clasificacion.ipynb
└─ regresion/
   └─ regresion.ipynb
```

---

## 1) Clasificación: Fatiga muscular con EMG

Construir un sistema que clasifique fatiga vs no fatiga a partir de señales electromiográficas (**EMG**).

### Dataset
Se utiliza el dataset de Hugging Face:
- `YominE/Muscle_Fatigue_Cycling`

### Flujo
- Carga del dataset desde `datasets.load_dataset(...)`
- Segmentación de la señal en **ventanas de 1 segundo** (fs = 1000 Hz)
- Extracción de características por canal (ejemplos):
  - RMS
  - Varianza
  - Zero Crossing Rate (ZCR)
  - Mean Absolute Value (MAV)
  - Potencia espectral (Welch)
  - Frecuencia media y mediana
- EDA (histogramas, correlación, boxplots, etc.)
- Entrenamiento y evaluación de modelos:
  - kNN
  - Decision Tree
  - Random Forest (+ GridSearchCV)
  - Gradient Boosting
- Implementación de una **DNN (Keras/TensorFlow)** con Dropout
- Métricas usadas: Accuracy, Precision, Recall y F1-score

---

## 2) Regresión: Predicción de edad con CNN (UTKFace)

Entrenar un modelo de **regresión** para predecir la **edad** a partir de imágenes faciales.

### Dataset (requerido)
Para ejecutar este notebook necesitas descargar el dataset desde Kaggle:

- Kaggle – [arashnic/faces-age-detection-dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset)

> El notebook asume una carpeta local con las imágenes (por ejemplo `dataset/UTKFace/`) y lee archivos `.jpg`.  
> La edad se extrae desde el nombre del archivo (formato típico: `edad_genero_raza_....jpg`).

### Flujo
- Carga de imágenes con OpenCV (`cv2`)
- Resize a `128x128`
- Normalización de pixeles a rango `[0, 1]`
- Train/Test split
- Modelo CNN en TensorFlow/Keras:
  - Conv2D + MaxPooling
  - Flatten + Dense + Dropout
  - Salida continua (edad)
- Entrenamiento y evaluación:
  - `loss = mse`
  - `metric = mae`
  - Se reporta **MSE**, **MAE** y **R²**

---

## Requisitos

- Python 3.x
- numpy, pandas, matplotlib, seaborn
- scikit-learn
- scipy
- tensorflow / keras
- datasets (Hugging Face)
- opencv-python (para el notebook de regresión)

---

## Cómo ejecutar

### Opción A) Local (recomendado)
1. Clona el repositorio:
   ```bash
   git clone https://github.com/JeronimoRodriguezRestrepo/Workshop_2_IA.git
   cd Workshop_2_IA
   ```

2. Instala dependencias (mínimas):
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy tensorflow datasets opencv-python
   ```

3. Abre Jupyter:
   ```bash
   jupyter notebook
   ```

4. Ejecuta los notebooks:
   - `clasificacion/clasificacion.ipynb`
   - `regresion/regresion.ipynb`

### Nota importante (Regresión)
El notebook de regresión espera una estructura como:
```text
regresion/
└─ dataset/
   └─ UTKFace/
      ├─ 25_0_0_....jpg
      ├─ 32_1_3_....jpg
      └─ ...
```
Si tu dataset está en otra ruta, actualiza la variable `ruta` en el notebook.

---

## Resultados
Los resultados (métricas y gráficas) se generan directamente al correr cada notebook.

---

