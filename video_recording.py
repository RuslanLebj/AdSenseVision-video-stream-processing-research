# Импорт необходимых модулей
import cv2


# Замените следующие значения на данные вашей IP-камеры
ip_address = '192.168.0.28'
username = 'admin'
password = '228Froggit322'

# Формируем URL для подключения к IP-камере
url = f'rtsp://{username}:{password}@{ip_address}/Streaming/Channels/1'

# Подключение к IP-камере
cap = cv2.VideoCapture(url)

# Получение ширины и высоты видеопотока
width = int(cap.get(3))
height = int(cap.get(4))

# Настройка объекта VideoWriter для записи в файл
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Выберите кодек и параметры
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))  # 'output.avi' - имя выходного файла


while True:

    # Считываем кадр из видеопотока
    ret, frame = cap.read()

    if not ret:
        print("Не удалось прочитать кадр")
        break


    # Создаем окно с поддержкой изменения размеров
    cv2.namedWindow('IP Camera: Detected Faces and Eyes', cv2.WINDOW_NORMAL)

    # Отображение результата
    cv2.imshow('IP Camera: Detected Faces and Eyes', frame)

    # Запись кадра в видеофайл
    out.write(frame)

    # Для выхода из цикла нажать ESC
    key = cv2.waitKey(1)
    if key == 27:
        break

# Освобождаем ресурсы и закрываем окна
cap.release()
