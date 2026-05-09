import cv2
from ultralytics import YOLO

# -- Configuración --
RUTA_MODELO   = 'runs/detect/experimento_13/weights/best.pt'
CONFIANZA_MIN = 0.4
CAMARA        = 0

traducciones = {
    'Scalpel':                    'Bisturi',
    'Straight_Dissection_Clamp':  'Pinza Recta',
    'Straight_Mayo_Scissor':      'Tijera Recta Mayo',
    'Curved_Mayo_Scissor':        'Tijera Curva Mayo'
}

# Instrumentos que deben estar presentes en la bandeja
BANDEJA_ESPERADA = {'Bisturi', 'Pinza Recta', 'Tijera Recta Mayo', 'Tijera Curva Mayo'}

# Colores BGR
COLOR_COMPLETA   = (0, 200, 0)
COLOR_INCOMPLETA = (0, 0, 220)
COLOR_CAJA       = (0, 165, 255)

def verificar_bandeja(clases_detectadas: set) -> tuple[str, tuple]:
    faltantes = BANDEJA_ESPERADA - clases_detectadas
    if not faltantes:
        return 'BANDEJA COMPLETA', COLOR_COMPLETA
    elementos = ', '.join(faltantes)
    return f'FALTA: {elementos}', COLOR_INCOMPLETA

def dibujar_detecciones(frame, resultado, nombres_clases):
    for caja in resultado.boxes:
        x1, y1, x2, y2 = map(int, caja.xyxy[0])
        clase_idx    = int(caja.cls[0])
        confianza    = float(caja.conf[0])
        nombre_clase = nombres_clases[clase_idx]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_CAJA, 2)
        etiqueta = f'{nombre_clase} {confianza:.2f}'
        (ancho_txt, alto_txt), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - alto_txt - 6), (x1 + ancho_txt, y1), COLOR_CAJA, -1)
        cv2.putText(frame, etiqueta, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame

def main():
    modelo = YOLO(RUTA_MODELO)
    
    # 1. Obtener nombres originales del modelo
    nombres_originales = modelo.names 
    
    # 2. Traducir nombres y crear un nuevo diccionario
    nombres_clases = {idx: traducciones.get(nombre, nombre) for idx, nombre in nombres_originales.items()}

    camara = cv2.VideoCapture(CAMARA)
    if not camara.isOpened():
        print('Error: no se pudo abrir la camara.')
        return

    while True:
        ret, frame = camara.read()
        if not ret: break

        resultados = modelo.predict(frame, conf=CONFIANZA_MIN, verbose=False)
        resultado  = resultados[0]

        # Usar los nombres ya traducidos para la lógica de la bandeja
        clases_detectadas = set(nombres_clases[int(c)] for c in resultado.boxes.cls)
        estado, color_estado = verificar_bandeja(clases_detectadas)

        frame = dibujar_detecciones(frame, resultado, nombres_clases)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color_estado, -1)
        cv2.putText(frame, estado, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        cv2.imshow('Deteccion de Instrumentos', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    camara.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
