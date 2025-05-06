import cv2
import numpy as np
import pyodbc
from ultralytics import YOLO
from collections import deque
from datetime import datetime

# Configura tu conexión SQL Server


conn = pyodbc.connect(
    f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
)
cursor = conn.cursor()

# Cargar modelo YOLOv8
model = YOLO("yolov8n.pt")

# Línea de conteo
LINE_X = 400
entradas = 0
track_history = {}

# Usa la cámara con DirectShow para evitar errores MSMF
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Verifica que la cámara se abra correctamente
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Establece resolución
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("No se pudo capturar el frame")
#         break

#     results = model.track(frame, persist=True, classes=[0])
#     annotated_frame = results[0].plot()

#     if results[0].boxes.id is not None:
#         ids = results[0].boxes.id.cpu().numpy().astype(int)
#         boxes = results[0].boxes.xyxy.cpu().numpy()

#         for i, box in enumerate(boxes):
#             person_id = ids[i]
#             x1, y1, x2, y2 = box.astype(int)
#             cx = int((x1 + x2) / 2)

#             # Dibuja el punto central
#             cv2.circle(annotated_frame, (cx, y1), 4, (0, 255, 0), -1)

#             # Control de historial para evitar doble conteo
#             if person_id not in track_history:
#                 track_history[person_id] = deque(maxlen=2)
#             track_history[person_id].append(cx)

#             if len(track_history[person_id]) == 2:
#                 if track_history[person_id][0] < LINE_X <= track_history[person_id][1]:
#                     entradas += 1
#                     cursor.execute("INSERT INTO Entradas (contador, fecha) VALUES (?, ?)", entradas, datetime.now())
#                     conn.commit()
#                     track_history[person_id].clear()

#     # Dibuja la línea y el contador
#     cv2.line(annotated_frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (255, 0, 0), 2)
#     cv2.putText(annotated_frame, f'Entradas: {entradas}', (20, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     cv2.imshow("Conteo de Entradas", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Limpieza
# cap.release()
# conn.close()
# cv2.destroyAllWindows()
# ... todo igual hasta antes del while
personas_contadas = set()  # <<--- NUEVO

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame")
        break

    results = model.track(frame, persist=True, classes=[0])
    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            person_id = ids[i]
            if person_id in personas_contadas:
                continue  # Ya fue contado, lo ignoramos

            x1, y1, x2, y2 = box.astype(int)
            cx = int((x1 + x2) / 2)

            # Dibuja el punto central
            cv2.circle(annotated_frame, (cx, y1), 4, (0, 255, 0), -1)

            if person_id not in track_history:
                track_history[person_id] = deque(maxlen=2)
            track_history[person_id].append(cx)

            if len(track_history[person_id]) == 2:
                if track_history[person_id][0] < LINE_X <= track_history[person_id][1]:
                    entradas += 1
                    personas_contadas.add(person_id)  # <<--- Agrega el ID ya contado
                    cursor.execute("INSERT INTO Entradas (contador, fecha) VALUES (?, ?)", entradas, datetime.now())
                    conn.commit()
                    track_history[person_id].clear()

    # Dibuja línea y contador
    cv2.line(annotated_frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (255, 0, 0), 2)
    cv2.putText(annotated_frame, f'Entradas: {entradas}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Conteo de Entradas", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
conn.close()
cv2.destroyAllWindows()
