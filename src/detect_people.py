import cv2
import ultralytics
from ultralytics import YOLO
from tqdm import tqdm
import subprocess

def detect_people_in_video(input_path: str, output_path: str, confidence: float):
    """
    Обнаруживает людей на видео и сохраняет результат в выходной файл.

    Args:
        input_path (str): Путь к входному видеофайлу.
        output_path (str): Путь к выходному видеофайлу.
        confidence (float): Пороговый уровень уверенности модели.
    """
    # Загрузка предобученной модели YOLOv8
    model = YOLO("yolov8s.pt")  # 's' означает small модель

    # Открытие видеофайла
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {input_path}")

    # Получение параметров видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Создание VideoWriter для записи результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Обработка кадров
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), desc="Обработка кадров"):
        ret, frame = cap.read()
        if not ret:
            break
        # Детекция объектов
        results = model(frame, classes=[0], conf=confidence)  # <-- фильтрация по классу "person", уверенность > confidence
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    # Освобождение ресурсов
    cap.release()
    out.release()
    print(f"Видео успешно сохранено: {output_path}")

if __name__ == "__main__":
    INPUT_VIDEO_PATH = "data/crowd.mp4"
    OUTPUT_VIDEO_PATH = "results/output_crowd_detected_person_05.mp4"
    FINAL_OUTPUT = "results/output_optimized.mp4"

    detect_people_in_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, 0.5)

    print("Сжатие видео...")
    subprocess.run([
        'ffmpeg',
        '-i', OUTPUT_VIDEO_PATH,
        '-vcodec', 'libx264',
        '-crf', '28',
        FINAL_OUTPUT
    ])

    print(f"Финальное видео готово: {FINAL_OUTPUT}")
