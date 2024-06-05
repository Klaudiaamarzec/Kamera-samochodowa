from tkinter import filedialog
import pygame
import os
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
    edges = cv.Canny(blurred, 30, 150)  # 50, 150
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

    for (x, y, w, h) in bodies_detection:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
    main()


def browse_file(video_path):
    if video_path:
        detect_objects_with_switch(video_path)


def get_video_list():
    videos_path = "Videos"
    video_files = [f for f in os.listdir(videos_path) if f.endswith(".mp4")]
    return video_files


def draw_start_screen(screen, video_files):
    pink = (255, 192, 203)
    yellow = (255, 255, 0)
    black = (0, 0, 0)

    font = pygame.font.SysFont(None, 36)

    screen.fill(pink)

    choose_file_text = font.render("WYBIERZ PLIK", True, black)
    choose_file_text_rect = choose_file_text.get_rect(center=(700 / 2, 50))
    screen.blit(choose_file_text, choose_file_text_rect)

    # Tworzenie rozwijanej listy
    for i, video in enumerate(video_files):
        text_surface = font.render(video, True, black)
        text_rect = text_surface.get_rect(topleft=(120, 100 + i * 30))
        if text_rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(screen, yellow, text_rect)
        screen.blit(text_surface, text_rect)

    legend_texts = [
        "KLAWISZOLOGIA",
        " ",
        "Spacja - wyłączenie nagrania",
        "O - wykrywanie słupków",
        "P - wykrywanie ludzi",
        "C - wykrywanie samochodów",
        "A - wykrywanie wszystkiego na raz"
    ]

    legend_y = 100 + len(video_files) * 30 + 20

    legend_width = 450
    legend_height = len(legend_texts) * 30 + 20
    legend_rect = pygame.Rect(120, legend_y, legend_width, legend_height)
    pygame.draw.rect(screen, yellow, legend_rect)

    for legend_text in legend_texts:
        legend_surface = font.render(legend_text, True, black)
        legend_rect = legend_surface.get_rect(topleft=(120, legend_y))
        screen.blit(legend_surface, legend_rect)
        legend_y += 30  # Odstęp między kolejnymi liniami w legendzie

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if 200 <= mouse_pos[0] <= 400:
                for i, video in enumerate(video_files):
                    text_surface = font.render(video, True, yellow)
                    text_rect = text_surface.get_rect(topleft=(200, 100 + i * 30))
                    if text_rect.collidepoint(mouse_pos):
                        return os.path.join("Videos", video)

    return None


def main():
    pygame.init()

    screen_width = 700
    screen_height = 550
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Konfigurator")

    video_files = get_video_list()
    video_path = None

    while video_path is None:
        video_path = draw_start_screen(screen, video_files)
        pygame.display.flip()

    pygame.quit()
    browse_file(video_path)


if __name__ == "__main__":
    main()
