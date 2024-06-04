import cv2
import cv2 as cv
import numpy as np


car_cascade = cv.CascadeClassifier('haarcascade_car.xml')
people_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_fullbody.xml')


def detect_objects(frame, mode, bumper):

    cv.drawContours(frame, bumper, -1, (0, 255, 0), 2)

    if mode == 'obstacles':
        # Wykrywaj przeszkody na drodze
        processed_frame = detect_bars(frame)
    elif mode == 'people':
        # Wykrywaj ludzi
        processed_frame = detect_people(frame)
    elif mode == 'cars':
        # Wykrywaj samochody
        processed_frame = detect_cars(frame)
    elif mode == 'all':
        processed_frame = detect_bars(frame)
        processed_frame = detect_people(processed_frame)
        processed_frame = detect_cars(processed_frame)
    else:
        processed_frame = frame
    return processed_frame


def detect_bumper(frame):
    # Kod detekcji przeszkód na drodze (bez zmian)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)  # 50, 150

    height, width = frame.shape[:2]
    lower_half_edges = edges[height // 2:height, :]

    # Kontury na obrazie
    contours, _ = cv.findContours(lower_half_edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Przesunięcie konturów do ich właściwej pozycji w oryginalnym obrazie
        contour[:, :, 1] += height // 2

    # Sortowanie według długości obwodu (malejąco)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    # 8 najdłuższych konturów
    longest_contours = sorted_contours[:8]

    bumper = []
    for contour in longest_contours:
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = float(w) / h
        # Jeśli stosunek szerokości do wysokości jest w zakresie, dodaj kontur
        if aspect_ratio > 2.5:
            bumper.append(contour)
            break

    return bumper


def detect_bars(frame):
    # Kod detekcji przeszkód na drodze (bez zmian)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 30, 150)   # 50, 150
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=320, maxLineGap=70)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 67 < angle < 97:
                    cv.rectangle(frame, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (0, 255, 255), 2)
                    break
    return frame


def detect_people(frame):
    bodies_detection = people_cascade.detectMultiScale(frame, 1.18, 3)

    for (x,y,w,h) in bodies_detection:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    return frame


def detect_cars(frame):
    cars_detection = car_cascade.detectMultiScale(frame, 1.4, 2)

    for (x, y, w, h) in cars_detection:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return frame


def detect_objects_with_switch(video_path):
    cap = cv.VideoCapture(video_path)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    resized_width = int(width * 0.5)
    resized_height = int(height * 0.5)
    mode = 'obstacles'  # Początkowy tryb detekcji

    # Znajdź zderzak
    ret, frame = cap.read()
    if not ret:
        return
    bumper = detect_bumper(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = detect_objects(frame, mode, bumper)
        if processed_frame is not None:
            resized_frame = cv.resize(processed_frame, (resized_width, resized_height))
            cv.imshow('Detection', resized_frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            mode = 'obstacles'
        elif key == ord('p'):
            mode = 'people'
        elif key == ord('c'):
            mode = 'cars'
        elif key == ord('a'):
            mode = 'all'
        elif key == ord(' '):  # Obsługa zamknięcia obrazu po naciśnięciu spacji
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_path = "Videos/Parkowanie - samochody.mp4"
    detect_objects_with_switch(video_path)
