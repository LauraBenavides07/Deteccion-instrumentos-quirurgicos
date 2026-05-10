# Sistema de Deteccion Automatizada de Instrumentos Quirurgicos

Proyecto desarrollado para el programa Talento Tech 2026 - curso: Inteligencia Artificial nivel innovador.
Detecta instrumentos quirúrgicos en tiempo real mediante visión por computador usando YOLOv8, donde verifica si la bandeja quirúrgica esta completa antes o después de un procedimiento.

---

## Clases detectadas

| ID | Instrumento |
|----|-------------|
| 0  | Bisturi (Scalpel) |
| 1  | Pinza de Disección (Straight Dissection Clamp) |
| 2  | Tijera Recta Mayo (Straight Mayo Scissor) |
| 3  | Tijera Curva Mayo (Curved Mayo Scissor) |

---

## Requisitos

- Python 3.10.11
- VS Code
- Camara web (para inferencia en tiempo real)

---

## Dataset

El proyecto usa el dataset **Labeled Surgical Tools and Images** disponible en Kaggle.

- Enlace: https://www.kaggle.com/datasets/dilavado/labeled-surgical-tools
- Total de imagenes: 3.009
- Clases: 4 instrumentos quirurgicos

## Instalacion

1. Clona el repositorio:
```bash
git clone https://github.com/TuUsuario/NombreRepositorio.git
cd NombreRepositorio
```

2. Crea el entorno virtual e instala dependencias de entrenamiento:
```bash
python -m venv _env
_env\Scripts\activate
pip install -r requirements_train.txt
```

3. Descarga el dataset desde Kaggle y coloca las carpetas `Images/` y `Labels/`
   dentro de una carpeta llamada `data/` en la raiz del proyecto.

---

## Estructura del proyecto

```
proyecto/
├── data/                          <- dataset original
│   ├── Images/
│   └── Labels/
├── dataset_yolo/                  <- generado por el notebook 1
├── test_images/                   <- imagenes de prueba para inferencia
├── runs/                          <- resultados del entrenamiento
├── classes.txt                    <- nombres de las clases
├── data.yaml                      <- configuracion del dataset para YOLOv8
├── generate_dataset_yolo.ipynb    <- notebook 1: organiza el dataset
├── experimentation_yolo.ipynb     <- notebook 2: entrena y evalua el modelo
├── time_real.py                   <- inferencia en tiempo real con camara
├── requirements_train.txt         <- dependencias para entrenar
└── requirements_time_real.txt     <- dependencias para inferencia
```

---

## Uso

### Paso 1 - Organizar el dataset

Abre y ejecuta todas las celdas de `generate_dataset_yolo.ipynb`.
Este notebook recorre las subcarpetas del dataset, empareja cada imagen con su label
y divide los datos en 80% train y 20% val.

### Paso 2 - Entrenar el modelo

Abre y ejecuta todas las celdas de `experimentation_yolo.ipynb`.
El modelo entrenado se guarda en `runs/detect/experimento_1/weights/best.pt`.

Parametros usados:
- Modelo base: yolov8n.pt
- Epocas: 30
- Batch: 8
- Tamano de imagen: 320

### Paso 3 - Inferencia en tiempo real

Instala las dependencias de inferencia y ejecuta el script:
```bash
pip install -r requirements_time_real.txt
python time_real.py
```

Apunta la cámara a los instrumentos quirúrgicos. El sistema mostrará:
- Una caja alrededor de cada instrumento detectado con su nombre en español y nivel de confianza.
- Barra verde en la parte superior, si la bandeja está completa.
- Barra roja indicando que instrumentos faltan si la bandeja esta incompleta.

Presiona `Q` para salir.

---

## Autor
 
- Laura Isabel Benavides Pito
Talento Tech 2026 - MINTIC | Inteligencia Artificial | Nivel Innovador | Popayan, Cauca
