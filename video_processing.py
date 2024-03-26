# Импорт необходимых модулей
import cv2
import numpy as np
import dlib

# Замените следующие значения на данные вашей IP-камеры
ip_address = '192.168.0.28'
username = 'admin'
password = '228Froggit322'

# Формируем URL для подключения к IP-камере
url = f'rtsp://{username}:{password}@{ip_address}/Streaming/Channels/1'

# Подключение к Веб-камере
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Путь к вашему файлу AVI
video_path = 'output.avi'

# Подключение к сохраненному видеопотоку
cap = cv2.VideoCapture(video_path)

# Подключение к IP-камере
# cap = cv2.VideoCapture(url)

# Получение ширины и высоты видеопотока
width = int(cap.get(3))
height = int(cap.get(4))

# Размеры нового разрешения (можете настроить по вашим требованиям)
new_width = 640
new_height = 480

# Загрузка предварительно обученной модели Dlib для обнаружения лиц
detector = dlib.get_frontal_face_detector()

# Загрузка предварительно обученной модели для обнаружения ключевых точек лица
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")


# Настройка объекта VideoWriter для записи в файл
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Выберите кодек и параметры
# out = cv2.VideoWriter('output_processed.avi', fourcc, 20.0, (width, height))  # 'output.avi' - имя выходного файла


def draw_line(frame, a, b, color=(255, 255, 0)):
    cv2.line(frame, a, b, color, 10)


while True:
    # Считываем кадр из видеопотока
    ret, frame = cap.read()

    if ret:

        # Сжатие изображения (уменьшение разрешения)
        # frame = cv2.resize(frame, (new_width, new_height))

        # Преобразование кадра в черно-белый формат (эффективнее для обнаружения лиц)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц
        faces = detector(grayFrame)

        # Обход списка всех лиц попавших на изображение
        for face in faces:
            # Получение ключевых точек лица
            landmarks = predictor(grayFrame, face)

            # Горизотнальное направление:
            # Вычисление расстояния между крайней левой точкой лица и левым глазом (Евклидово расстояние)
            left_eye_distance = np.linalg.norm(
                np.array([landmarks.part(0).x, landmarks.part(0).y]) - np.array(
                    [landmarks.part(36).x, landmarks.part(36).y]))

            # Вычисление расстояния между крайней правой точкой лица и правым глазом (Евклидово расстояние)
            right_eye_distance = np.linalg.norm(
                np.array([landmarks.part(16).x, landmarks.part(16).y]) - np.array(
                    [landmarks.part(45).x, landmarks.part(45).y]))

            # Задаем порог для определения направления взгляда в центр по горизонтали в виде % от ср. расстояния
            # *Это значение может потребоваться настроить в зависимости от конкретных условий
            threshold_horizontal_coef = 0.3
            threshold_horizontal = ((left_eye_distance + right_eye_distance) / 2) * threshold_horizontal_coef

            # Определение направления области взгляда по горизонтали
            if abs(left_eye_distance - right_eye_distance) < threshold_horizontal:
                gaze_direction_horizontal = "Middle"
            elif left_eye_distance < right_eye_distance:
                gaze_direction_horizontal = "Left"
            else:
                gaze_direction_horizontal = "Right"

            # Вертикальное направление:
            # Нахождение точки между бровями (внутренними точками бровей) - лобная точка
            point_between_eyebrows = (
                (landmarks.part(21).x + landmarks.part(22).x) // 2, (landmarks.part(21).y + landmarks.part(22).y) // 2)

            # Вычисление расстояния между лобной точкой и точкой на кончике носа (Евклидово расстояние)
            forehead_nose_distance = np.linalg.norm(
                np.array([point_between_eyebrows[0], point_between_eyebrows[1]]) - np.array(
                    [landmarks.part(30).x, landmarks.part(30).y]))

            # Вычисление расстояния между точкой на подбородке и точкой на кончике носа (Евклидово расстояние)
            chin_nose_distance = np.linalg.norm(
                np.array([landmarks.part(8).x, landmarks.part(8).y]) - np.array(
                    [landmarks.part(30).x, landmarks.part(30).y]))

            # Задаем порог для определения направления взгляда в центр по горизонтали в виде % от ср. расстояния
            # *Это значение может потребоваться настроить в зависимости от конкретных условий
            threshold_vertical_coef = 0.1
            threshold_vertical = ((forehead_nose_distance + chin_nose_distance) / 2) * threshold_vertical_coef

            # Определение вертикального направления области взгляда
            if abs(forehead_nose_distance - chin_nose_distance) < threshold_vertical:
                gaze_direction_vertical = "Middle"
            elif forehead_nose_distance < chin_nose_distance:
                gaze_direction_vertical = "Top"
            else:
                gaze_direction_vertical = "Bottom"

            # Визуализация:
            # Вывод направления
            cv2.putText(frame, gaze_direction_vertical + " " + gaze_direction_horizontal, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Получение координат контрольных точек и их построение на изображении
            landmarks = predictor(grayFrame, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

            # Визуализация точек для определения горизонтального направления
            cv2.circle(frame, (landmarks.part(0).x, landmarks.part(0).y), 3, (0, 155, 0), -1)
            cv2.circle(frame, (landmarks.part(36).x, landmarks.part(36).y), 3, (0, 255, 0), -1)

            cv2.circle(frame, (landmarks.part(16).x, landmarks.part(16).y), 3, (0, 0, 155), -1)
            cv2.circle(frame, (landmarks.part(45).x, landmarks.part(45).y), 3, (0, 0, 255), -1)

            # Визуализация точек для определения вертикального направления
            cv2.circle(frame, (point_between_eyebrows[0], point_between_eyebrows[1]), 3, (0, 255, 255), -1)
            cv2.circle(frame, (landmarks.part(30).x, landmarks.part(30).y), 3, (0, 255, 255), -1)
            cv2.circle(frame, (landmarks.part(8).x, landmarks.part(8).y), 3, (0, 255, 255), -1)

            # Получение координат вершин прямоугольника и его построение на изображении
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Создаем окно с поддержкой изменения размеров
    cv2.namedWindow('IP Camera: Detected Faces and Eyes', cv2.WINDOW_NORMAL)

    # Отображение результата
    cv2.imshow('IP Camera: Detected Faces and Eyes', frame)

    # Запись кадра в видеофайл
    # out.write(frame)

    # Для выхода из цикла нажать ESC
    key = cv2.waitKey(1)
    if key == 27:
        break

# Освобождаем ресурсы и закрываем окна
cap.release()
