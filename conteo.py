import cv2
import numpy as np
import pyodbc
from ultralytics import YOLO
from datetime import datetime

#Conexion a la base de datos
server = 
database = 
username = 
password = 


conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
cursor = conn.cursor()


model = YOLO("yolov8n.pt")

LINE_X = 400
entradas = 0

personas_contadas = set()

cap = cv2.VideoCapture('C:/Users/Programdor Junior 2/Downloads/video_prueba.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0])
    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            person_id = ids[i]
            x1, y1, x2, y2 = box.astype(int)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 0), -1)

            if person_id not in personas_contadas:
                entradas += 1
                personas_contadas.add(person_id)

                cursor.execute("INSERT INTO Entradas (contador, fecha) VALUES (?, ?)", entradas, datetime.now())
                conn.commit()

    cv2.line(annotated_frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (255, 0, 0), 2)
    cv2.putText(annotated_frame, f'Entradas: {entradas}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Conteo de Entradas", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import pyodbc
# from ultralytics import YOLO
# from collections import deque
# from datetime import datetime

# # Configuración de la conexión a SQL Server


# # Establecer la conexión
# conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
# cursor = conn.cursor()

# # Cargar modelo YOLOv8 preentrenado
# model = YOLO("yolov8n.pt")  # Usa 'yolov8s.pt' si quieres más precisión

# # Línea virtual horizontal (Y)
# #LINE_Y = 210
# LINE_X = 400
# entradas = 0

# # Trackers simples por ID
# track_history = {}

# # Captura de video
# #cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('C:/Users/Programdor Junior 2/Downloads/video_prueba.mp4')

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model.track(frame, persist=True, classes=[0])  # clase 0 = persona
#     annotated_frame = results[0].plot()

#     # Extraer información de seguimiento
#     if results[0].boxes.id is not None:
#         ids = results[0].boxes.id.cpu().numpy().astype(int)
#         boxes = results[0].boxes.xyxy.cpu().numpy()

#         for i, box in enumerate(boxes):
#             person_id = ids[i]
#             x1, y1, x2, y2 = box.astype(int)
#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)

#             # Dibujar centro
#             cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 0), -1)

#             # Seguimiento del historial de posición
#             if person_id not in track_history:
#                 track_history[person_id] = deque(maxlen=2)
#             track_history[person_id].append(cx)
#             #track_history[person_id].append(cy)

#             # Verificar si cruzó la línea de arriba hacia abajo (entrada)
#             if len(track_history[person_id]) == 2:
#                 if track_history[person_id][0] < track_history[person_id][1]: #LINE_Y <= track_history[person_id][1]:
#                     entradas += 1
#                     # Guardar en la base de datos
#                     cursor.execute("INSERT INTO Entradas (contador, fecha) VALUES (?, ?)", entradas, datetime.now())
#                     conn.commit()

#                     # Limpiar historial para evitar doble conteo
#                     track_history[person_id].clear()

#     # Dibujar línea y contador
#     #cv2.line(annotated_frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (255, 0, 0), 2)
#     cv2.line(annotated_frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (255, 0, 0), 2)
#     cv2.putText(annotated_frame, f'Entradas: {entradas}', (20, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     cv2.imshow("Conteo de Entradas", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cerrar la conexión
# cap.release()
# conn.close()
# cv2.destroyAllWindows()
