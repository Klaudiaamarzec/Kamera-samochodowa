import cv2 as cv
import numpy as np


def detect_parking_obstacles(video_path):
    # Ustaw kamerę
    cap = cv.VideoCapture(video_path)

    # Pobierz rozmiar klatki wideo
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # Ustaw nowy rozmiar okna
    resized_width = int(width * 0.7)
    resized_height = int(height * 0.7)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Wywołaj funkcję detekcji przeszkód
        obstacle_image = detect_bars(frame)

        # Zmniejsz rozmiar klatki obrazu
        resized_frame = cv.resize(obstacle_image, (resized_width, resized_height))
        # Wyświetl obraz
        cv.imshow('Obstacles Detected', resized_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # Zwolnij kamerę i zakończ program
    cap.release()
    cv.destroyAllWindows()


def detect_bars(frame):
    # Konwertuj obraz na odcienie szarości
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Wykonaj wygładzanie obrazu za pomocą filtru Gaussa
    blurred = cv.GaussianBlur(gray, (5, 5), 0)  # ksize 5x5
    # Wykryj krawędzie na obrazie za pomocą algorytmu Canny'ego
    edges = cv.Canny(blurred, 50, 150)  # 30, 150
    # Znajdź kontury na obrazie
    contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Zielony kontury na oryginalnym obrazie
    #cv.drawContours(frame, contours, -1, (0, 255, 0), 2)

    #Transformacja Hough'a do linii
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=75, maxLineGap=10)

    # Wykryj słupki i narysuj wokół nich żółte prostokąty
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Oblicz kąt nachylenia linii
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                # # Sprawdź, czy kąt nachylenia wskazuje na pionową linię
                # if 80 < angle < 100 or -100 < angle < -80:

                # Sprawdź, czy kąt nachylenia wskazuje na pionową lub lekko odchyloną linię
                if 70 < angle < 110 or -110 < angle < -70:
                    cv.rectangle(frame, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (0, 255, 255), 2)
                # Wykrywa słupki pod katem ale wykrywa tez inne elementy
                # elif 50 < angle < 70 or -70 < angle < -50:
                #     cv.rectangle(frame, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (0, 255, 255), 2)

    return frame


def detect_bumper(frame):
    # Konwertuj obraz na odcienie szarości
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Zastosuj filtr dolnoprzepustowy
    lowpass = cv.blur(gray, (5, 5))
    # Wykonaj wygładzanie obrazu za pomocą filtru Gaussa
    blurred = cv.GaussianBlur(lowpass, (5, 5), 0)
    # Wykryj krawędzie na obrazie za pomocą algorytmu Canny'ego
    edges = cv.Canny(blurred, 50, 150)
    # Znajdź kontury na obrazie
    contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Narysuj kontury na oryginalnym obrazie
    # obstacle_image = frame.copy()
    # cv.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Wykrywanie zderzaka
    max_area = 0
    car_contour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            car_contour = contour

    # Zaznaczenie granic zderzaka na obrazie
    if car_contour is not None:
        cv.drawContours(frame, [car_contour], -1, (0, 0, 255), 2)

    return frame


if __name__ == "__main__":
    video_path = "Videos/Parkowanie - słupki2.mp4"
    detect_parking_obstacles(video_path)
