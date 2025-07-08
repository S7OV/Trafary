# Dynamic_SSIM_Analysis

import os
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def calculate_pair(args):
    """
    Параллельная функция для расчёта SSIM между двумя соседними кадрами.

    Параметры:
        args (tuple): (i, folder_path, resize_factor)

    Возвращает:
        tuple: (i+1, ssim_value)
    """
    i, folder_path, resize_factor = args
    try:
        loader = FrameLoader(resize_factor=resize_factor)
        calculator = SSIMCalculator()
        prev_img = loader.load(os.path.join(folder_path, f"frame_{i}.jpg"))
        curr_img = loader.load(os.path.join(folder_path, f"frame_{i + 1}.jpg"))
        return i + 1, calculator.color_ssim(prev_img, curr_img)
    except Exception as e:
        print(f"❌ Ошибка при обработке кадра {i}: {e}")
        return i + 1, float('nan')


class KeyframeDetector:
    """
    Класс для поиска ключевых кадров на основе SSIM.

    Аргументы:
        fps (int): частота кадров видео
        min_stable_duration (int): минимальная длина стабильного участка в кадрах
    """

    def __init__(self, fps, window_size_sec=10, min_stable_duration=30, mask_lector=False):
        """
        Вычисляет ключевые кадры на основе метрики SSIM.

        Параметры:
            fps: частота кадров видео
            window_size_sec: размер окна для скользящего среднего (в секундах)
            min_stable_duration: минимальная длина стабильного участка (в кадрах)
        """
        self.fps = fps
        self.window_size_sec = window_size_sec
        self.min_stable_duration = min_stable_duration
        self.mask_lector = mask_lector

    def detect_keyframes(self, df):
        """
        Находит ключевые кадры в стабильных участках видео на основе метрики SSIM.

        Ключевой кадр выбирается как середина длительного участка с высокой стабильностью изображения,
        что может указывать на важный фрейм (например, слайд или заставка).

        Параметры:
            df (pd.DataFrame): таблица с колонкой 'ssim', содержащая значения SSIM между соседними кадрами.
                              Должна быть отсортирована по времени.

        Возвращает:
            pd.DataFrame: DataFrame с найденными ключевыми кадрами. Содержит:
                - frame_num (int): номер кадра (индекс)
                - timestamp_sec (float): временная метка в секундах

        Пример возвращаемого значения:
            | frame_num | timestamp_sec |
            |-----------|---------------|
            | 100       | 3.33          |
            | 250       | 8.33          |
        """

        # --- Шаг 1: Извлечение значений SSIM ---
        # Берём только те значения SSIM, которые не являются NaN
        ssim_values = df['ssim'].dropna().values

        # Определяем размер окна для скользящего среднего — 1 секунд видео
        # --- Вычисление размера окна ---
        if self.mask_lector:
            window_size = int(self.fps * self.window_size_sec) # <-- окно в 2 раза меньше
        else:
            window_size = int(self.fps * self.window_size_sec)

        # Если нет данных для анализа — выводим предупреждение и возвращаем пустой DataFrame
        if len(ssim_values) == 0:
            print("⚠️ Нет данных для анализа SSIM")
            return pd.DataFrame(columns=['frame_num', 'timestamp_sec'])

        # --- Шаг 2: Расчёт скользящего среднего SSIM ---
        # Рассчитываем среднее значение SSIM за последние window_size кадров
        rolling_mean = pd.Series(ssim_values).rolling(window=window_size, min_periods=1).mean()
        #print ('rolling_mean ', rolling_mean)

        # --- Шаг 3: Создание динамического порога ---
        # --- Расчёт динамического порога ---
        if self.mask_lector:
            dynamic_threshold = rolling_mean # - np.std(rolling_mean)   # на кадрах без лектора увеличиваем динамический порог
            min_stable_duration_set = self.min_stable_duration / 2 # уменьшаем длину стабильного участка на кадрах без лектора
        else:
            dynamic_threshold = rolling_mean - np.std(rolling_mean)  # <-- среднее - σ
            min_stable_duration_set = self.min_stable_duration 

        # Заполняем первые кадры резервным порогом (0.95), так как rolling_mean ещё не рассчитан
        dynamic_threshold = dynamic_threshold.fillna(0.95)

        # Ограничиваем порог в диапазоне [0.5; 1.0], чтобы избежать аномалий
        dynamic_threshold = np.clip(dynamic_threshold, 0.5, 1.0)

        # --- Шаг 4: Поиск стабильных участков ---
        stable_regions = []
        in_stable = False   # Флаг: сейчас ли мы внутри стабильного участка?
        start_idx = None    # Индекс начала стабильного участка



        # Перебираем все строки исходной таблицы df
        for idx, row in df.iterrows():
            if pd.isna(row['ssim']):
                continue  # Пропускаем первый кадр, у которого нет пары

            current_ssim = row['ssim']  # Текущее значение SSIM
            # Берём динамический порог для текущего кадра (или 0.95, если его нет)
            current_threshold = dynamic_threshold[idx] if idx < len(dynamic_threshold) else 0.95

            # --- Логика обнаружения стабильных участков ---
            if not in_stable:
                # Если мы ещё не вошли в стабильный участок
                if current_ssim > current_threshold:
                    # Если SSIM выше порога — начало стабильного участка
                    start_idx = idx
                    in_stable = True
            else:
                # Если мы уже внутри стабильного участка
                if current_ssim > current_threshold:
                    # Участок продолжается
                    continue
                else:
                    # Участок закончился — проверяем его длину
                    duration = idx - start_idx

                    # Если длина участка >= минимальной — сохраняем его как стабильный
                    if duration >= min_stable_duration_set:
                        mid_frame = start_idx + duration // 2  # Середина участка
                        stable_regions.append({
                            'start': start_idx,
                            'end': idx - 1,
                            'mid_frame': mid_frame
                        })

                    # Сбрасываем состояние
                    in_stable = False

        # --- Шаг 5: Формируем список ключевых кадров ---
        keyframes = [
            {'frame_num': region['mid_frame'], 'timestamp_sec': region['mid_frame'] / self.fps}
            for region in stable_regions
        ]

        # --- Шаг 6: Возвращаем результат в виде DataFrame ---
        return pd.DataFrame(keyframes)


class VideoFrameExtractor:
    """
    Извлекает кадры из видео и сохраняет их в папку.

    Аргументы:
        resize_factor (float): коэффициент уменьшения размера кадра
    """

    def __init__(self, resize_factor=0.5):
        self.resize_factor = resize_factor

    def extract_frames(self, video_path, output_folder="frames"):
        """
        Извлекает кадры из видео и сохраняет как .jpg.

        Параметры:
            video_path (str): путь к видеофайлу
            output_folder (str): папка для сохранения кадров

        Возвращает:
            tuple: (output_folder, fps)
        """
        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        clip = VideoFileClip(video_path)
        digits = len(str(total_frames - 1))  # <-- вычисляем здесь

        frame_paths = []
        for i in tqdm(range(total_frames), desc="🖼️ Извлечение кадров"):
            frame = clip.get_frame(i / clip.fps)
            frame_image = Image.fromarray(frame)
            filename = f"frame_{i:0{digits}d}.jpg"  # <-- используем digits
            path = os.path.join(output_folder, filename)
            frame_image.save(path)
            frame_paths.append(path)

        print(f"✅ Сохранено кадров: {len(frame_paths)}")
        return output_folder, fps, digits  # <-- возвращаем digits


class FaceAnalyzer:
    """
    Анализ лица через MediaPipe Face Mesh. Используется для определения речи по губам.
    """

    UPPER_LIP_IDXS = [61, 185, 40, 39, 37, 267, 269, 270, 409, 291]
    LOWER_LIP_IDXS = [308, 415, 310, 311, 312, 13, 14, 15, 16]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )

    def get_lip_distance(self, landmarks, img_w, img_h):
        """
        Вычисляет вертикальное расстояние между губами.

        Параметры:
            landmarks: список точек лица
            img_w, img_h: размеры кадра

        Возвращает:
            float: среднее вертикальное расстояние
        """
        upper_points = [int(landmarks[idx].y * img_h) for idx in self.UPPER_LIP_IDXS if idx < len(landmarks)]
        lower_points = [int(landmarks[idx].y * img_h) for idx in self.LOWER_LIP_IDXS if idx < len(landmarks)]

        if not upper_points or not lower_points:
            return 0
        return abs(np.mean(upper_points) - np.mean(lower_points))


class BodyPoseAnalyzer:
    """
    Обнаружение тела через MediaPipe Pose.
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1)

    def get_body_bbox(self, landmarks, img_w, img_h, padding=0.3):
        """
        Получает bounding box по всему телу.

        Параметры:
            landmarks: точки тела
            img_w, img_h: размеры кадра
            padding: отступ

        Возвращает:
            tuple: (x_min, y_min, x_max, y_max)
        """
        xs = [lm.x * img_w for lm in landmarks.landmark]
        ys = [lm.y * img_h for lm in landmarks.landmark]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        pad_w = (x_max - x_min) * padding
        pad_h = (y_max - y_min) * padding

        x_min = max(0, int(x_min - pad_w))
        y_min = max(0, int(y_min - pad_h))
        x_max = min(img_w, int(x_max + pad_w))
        y_max = min(img_h, int(y_max + pad_h))

        return (x_min, y_min, x_max, y_max)


class SegmentationProcessor:
    """
    Закрашивание человека в ROI через SelfieSegmentation.
    """

    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

    def mask_lector_in_roi(self, roi):
        """
        Закрашивает человека фоном в регионе интереса.

        Параметры:
            roi: регион интереса (ROI)

        Возвращает:
            np.array: изменённый ROI
        """
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        seg_result = self.segmentation.process(roi_rgb)
        person_mask_roi = seg_result.segmentation_mask > 0.1
        background_mask = ~person_mask_roi
        background_color = cv2.mean(roi, mask=background_mask.astype(np.uint8) * 255)[:3]
        roi_clean = roi.copy()
        roi_clean[person_mask_roi] = background_color
        return roi_clean


class LectorMasker:
    """
    Маскировка лектора в видео.
    """

    def __init__(self):
        self.body_analyzer = BodyPoseAnalyzer()
        self.segmenter = SegmentationProcessor()
        self.face_analyzer = FaceAnalyzer()

    def process_video_save_frames(self, video_path, output_folder="frames_no_lector",
                                  buffer_size=10, movement_threshold=0.5):
        """
        Обрабатывает видео и маскирует лектора.

        Параметры:
            video_path: путь к видео
            output_folder: папка для сохранения кадров
            buffer_size: размер буфера для анализа движения
            movement_threshold: порог изменения положения губ

        Возвращает:
            tuple: (output_folder, fps)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Ошибка открытия видео")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        digits = len(str(total_frames - 1))  # <-- вычисляем digits
        frame_saver = FrameSaver(output_folder, digits=digits)  # <-- передаем digits

        pbar = tqdm(total=total_frames, desc="🖼️ Обработка кадров")
        lip_distances = []
        frame_idx = 0

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.body_analyzer.pose.process(rgb_frame)

            if pose_results.pose_landmarks:
                bbox = self.body_analyzer.get_body_bbox(pose_results.pose_landmarks, frame.shape[1], frame.shape[0])
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    roi = frame[y_min:y_max, x_min:x_max]
                    cleaned_roi = self.segmenter.mask_lector_in_roi(roi)
                    frame[y_min:y_max, x_min:x_max] = cleaned_roi

            face_results = self.face_analyzer.face_mesh.process(rgb_frame)
            lip_dist = 0

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    img_h, img_w, _ = frame.shape
                    lip_dist = self.face_analyzer.get_lip_distance(face_landmarks.landmark, img_w, img_h)

            lip_distances.append(lip_dist)
            if len(lip_distances) > buffer_size:
                lip_distances.pop(0)

            frame_saver.save_frame(frame, frame_idx)
            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()
        print(f"✅ Обработано и сохранено {frame_idx} кадров в '{output_folder}'")
        return output_folder, fps, digits


class FrameSaver:
    """
    Сохраняет кадры в формате frame_XXXXXX.jpg
    """
    def __init__(self, output_folder, digits):
        self.output_folder = output_folder
        self.digits = digits  # <-- добавляем параметр
        os.makedirs(self.output_folder, exist_ok=True)

    def save_frame(self, frame, frame_idx):
        filename = f"frame_{frame_idx:0{self.digits}d}.jpg"
        path = os.path.join(self.output_folder, filename)
        cv2.imwrite(path, frame)
        return path

class FrameLoader:

    def __init__(self, resize_factor=0.5, mask_lector=False):
        self.resize_factor = resize_factor
        self.mask_lector = mask_lector
        if mask_lector:
            from mediapipe import solutions
            self.mp_pose = solutions.pose
            self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1)

    def load(self, path):
        img = Image.open(path).convert("RGB")
        if self.resize_factor != 1.0:
            new_size = tuple(int(dim * self.resize_factor) for dim in img.size)
            img = img.resize(new_size)
        img_np = np.array(img)
        
        if self.mask_lector:
            # Преобразуем изображение в формат OpenCV
            rgb_frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            pose_results = self.pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                img_h, img_w, _ = img_np.shape
                # Получаем bounding box тела
                x_min = min(int(lm.x * img_w) for lm in pose_results.pose_landmarks.landmark)
                y_min = min(int(lm.y * img_h) for lm in pose_results.pose_landmarks.landmark)
                x_max = max(int(lm.x * img_w) for lm in pose_results.pose_landmarks.landmark)
                y_max = max(int(lm.y * img_h) for lm in pose_results.pose_landmarks.landmark)

                # Расширяем bounding box на 10% для полноты маски
                pad_w = (x_max - x_min) * 0.1
                pad_h = (y_max - y_min) * 0.1
                x_min = max(0, int(x_min - pad_w))
                y_min = max(0, int(y_min - pad_h))
                x_max = min(img_w, int(x_max + pad_w))
                y_max = min(img_h, int(y_max + pad_h))

                # Создаем маску для фона
                mask = np.ones((img_h, img_w), dtype=np.uint8)
                mask[y_min:y_max, x_min:x_max] = 0

                # Заполняем область лектора средним цветом фона
                channels = cv2.split(img_np)
                background_color = [int(cv2.mean(ch, mask=mask)[0]) for ch in channels]
                img_np[y_min:y_max, x_min:x_max] = background_color

        return img_np


class SSIMCalculator:
    """
    Расчёт цветового SSIM между двумя кадрами
    """

    @staticmethod
    def color_ssim(img1, img2, win_size=11, k1=0.01, k2=0.03):
        """
        Цветовой SSIM через OpenCV.

        Параметры:
            img1, img2: два кадра
            win_size: размер окна
            k1, k2: константы SSIM

        Возвращает:
            float: значение SSIM
        """
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        mu1 = cv2.blur(img1, (win_size, win_size))
        mu2 = cv2.blur(img2, (win_size, win_size))

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.blur(img1 ** 2, (win_size, win_size)) - mu1_sq
        sigma2_sq = cv2.blur(img2 ** 2, (win_size, win_size)) - mu2_sq
        sigma12 = cv2.blur(img1 * img2, (win_size, win_size)) - mu1_mu2

        C1 = (k1 * 255) ** 2
        C2 = (k2 * 255) ** 2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator
        return float(ssim_map.mean())


class FrameProcessor:
    """
    Подготовка списка кадров для анализа SSIM

    Аргументы:
        folder_path: путь к папке с кадрами
        loader: объект FrameLoader
        calculator: объект SSIMCalculator
    """

    def __init__(self, folder_path, loader, calculator):
        self.folder_path = str(folder_path)
        self.loader = loader
        self.calculator = calculator
        self.frame_files = []
        self.frame_paths = []

    def find_frames(self):
        """
        Считывает и сортирует кадры из папки.

        Вызывает ошибку, если кадры не найдены.
        """
        if not isinstance(self.folder_path, (str, bytes, os.PathLike)):
            raise ValueError(f"❌ Некорректный путь к папке: {type(self.folder_path)}")

        self.frame_files = sorted(
            [f for f in os.listdir(self.folder_path) if f.startswith("frame_") and f.endswith(".jpg")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        if not self.frame_files:
            raise FileNotFoundError(f"❌ Не найдено кадров в папке {self.folder_path}")

        self.frame_paths = [os.path.join(self.folder_path, f) for f in self.frame_files]


class SSIMAnalyzer:
    """
    Последовательный или параллельный анализ SSIM

    Аргументы:
        processor: объект FrameProcessor
    """

    def __init__(self, processor):
        self.processor = processor
        self.data = []
        self.ssim_values = []

    def run_sequential(self):
        """
        Последовательный расчёт SSIM между кадрами.
        """
        self.data = [{"frame": self.processor.frame_files[0], "ssim": np.nan}]
        prev_img = self.processor.loader.load(self.processor.frame_paths[0])

        for i in tqdm(range(1, len(self.processor.frame_paths)), desc="🧮 Расчёт SSIM (CPU)"):
            curr_img = self.processor.loader.load(self.processor.frame_paths[i])
            ssim_value = self.processor.calculator.color_ssim(prev_img, curr_img)
            self.data.append({"frame": self.processor.frame_files[i], "ssim": ssim_value})
            self.ssim_values.append(ssim_value)
            prev_img = curr_img

    def run_parallel(self, mp_context=None):
        """
        Параллельный расчёт SSIM.

        Параметры:
            mp_context: контекст многопроцессности (spawn)
        """
        executor_class = ProcessPoolExecutor
        if mp_context is None:
            executor = executor_class()
        else:
            executor = executor_class(mp_context=mp_context)

        frame_files = self.processor.frame_files
        folder_path = self.processor.folder_path
        resize_factor = self.processor.loader.resize_factor

        args_list = [(i, str(folder_path), resize_factor) for i in range(len(frame_files) - 1)]
        futures = [executor.submit(calculate_pair, args) for args in args_list]
        pbar = tqdm(total=len(futures), desc="🧮 Расчёт SSIM (CPU)")
        results = []

        for future in as_completed(futures):
            try:
                idx, value = future.result()
                results.append((idx, value))
            except Exception as e:
                print(f"❌ Ошибка в параллельном процессе: {e}")
            pbar.update(1)
        pbar.close()

        results.sort(key=lambda x: x[0])
        self.data = [{"frame": frame_files[0], "ssim": np.nan}]
        for idx, value in results:
            self.data.append({"frame": frame_files[idx], "ssim": value})
            self.ssim_values.append(value)

    def analyze(self, use_parallel=True, output_csv="ssim_results.csv", mp_context=None):
        """
        Основной метод анализа SSIM.

        Параметры:
            use_parallel: использовать параллельность?
            output_csv: путь к CSV
            mp_context: контекст multiprocessing

        Возвращает:
            pd.DataFrame: таблица с SSIM
        """
        self.processor.find_frames()
        if use_parallel:
            self.run_parallel(mp_context=mp_context)
        else:
            self.run_sequential()

        df = pd.DataFrame(self.data)
        df.to_csv(output_csv, index=False)
        print(f"💾 Результаты сохранены в {output_csv}")
        return df

class VideoSSIMPipeline:
    def __init__(self, video_path, mask_lector=False, resize_factor=0.5, use_parallel=True,
                 window_size_sec=10, min_stable_duration=30):
        """
        Основной класс для полного цикла анализа видео.

        Аргументы:
            video_path: путь к видеофайлу
            mask_lector: маскировать лектора?
            resize_factor: коэффициент уменьшения кадра
            use_parallel: использовать параллельную обработку?
            window_size_sec: длительность окна для скользящего среднего SSIM (в секундах)
            min_stable_duration: минимальная длина стабильного участка (в кадрах)
        """
        self.video_path = video_path
        self.mask_lector = mask_lector
        self.resize_factor = resize_factor
        self.use_parallel = use_parallel
        self.window_size_sec = window_size_sec
        self.min_stable_duration = min_stable_duration

    @staticmethod
    def _delete_temp_folder(folder_path):
        """Удаляет временную папку"""
        try:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"🗑️ Временная папка удалена: {folder_path}")
        except Exception as e:
            print(f"❌ Ошибка при удалении папки: {e}")

    def run(self, output_keyframes="keyframes.csv", generate_pdf_report=True,
            output_pdf="keyframes_report.pdf", cleanup=True, mp_context=None):
        """
        Выполняет полный цикл анализа видео:
            1. Извлечение или обработка кадров
            2. Расчёт SSIM
            3. Поиск ключевых кадров
            4. Сохранение CSV
            5. (Опционально) Создание PDF-отчёта
            6. Удаление временных файлов

        Параметры:
            output_keyframes (str): имя выходного CSV-файла
            generate_pdf_report (bool): генерировать ли PDF-отчёт
            output_pdf (str): имя PDF-файла
            cleanup (bool): удалять ли временные кадры после обработки
            mp_context (multiprocessing.context): контекст для параллельной обработки

        Возвращает:
            pd.DataFrame: таблица ключевых кадров
        """
        # --- Шаг 1: Извлечение или обработка кадров ---
        if self.mask_lector:
            lector_masker = LectorMasker()
            frames_folder, fps, digits = lector_masker.process_video_save_frames(self.video_path)
        else:
            extractor = VideoFrameExtractor(resize_factor=self.resize_factor)
            frames_folder, fps, digits = extractor.extract_frames(self.video_path)
            
        temp_frames_folder = frames_folder 
        # --- Шаг 2: Анализ SSIM с учетом маскировки ---
        loader = FrameLoader(
            resize_factor=self.resize_factor,
            mask_lector=self.mask_lector  # <-- Передаем флаг для маскировки
        )
        calculator = SSIMCalculator()
        frame_processor = FrameProcessor(
            folder_path=str(frames_folder),
            loader=loader,
            calculator=calculator
        )
        analyzer = SSIMAnalyzer(frame_processor)
        df = analyzer.analyze(use_parallel=self.use_parallel, output_csv=None, mp_context=mp_context)

        # --- Шаг 3: Поиск ключевых кадров ---
        detector = KeyframeDetector(
            fps=fps,
            window_size_sec=self.window_size_sec,
            min_stable_duration=self.min_stable_duration,
            mask_lector=self.mask_lector  # <-- теперь используется
        )

        keyframes_df = detector.detect_keyframes(df)  # <-- передаем digits

        # --- Шаг 4: Сохранение результатов ---
        if not keyframes_df.empty:
            keyframes_df.to_csv(output_keyframes, index=False)
            print(f"✅ Найдено ключевых кадров: {len(keyframes_df)}")
        else:
            print("❌ Ключевые кадры не найдены")

        # --- Шаг 5: Генерация PDF-отчёта (до удаления кадров!) ---
        if generate_pdf_report and not keyframes_df.empty:
            self.generate_report_pdf(keyframes_df, frames_folder, digits, video_path=self.video_path, output_pdf=output_pdf)

        # --- Шаг 6: Очистка временных файлов ---
        if cleanup:
            self._delete_temp_folder(temp_frames_folder)

        return keyframes_df



    def generate_report_pdf(self, keyframes_df, frames_folder, digits, video_path, output_pdf="keyframes_report.pdf"):
        """
        Генерирует PDF-отчёт с таблицей и превью ключевых кадров.
        Включает название видео в заголовке отчёта.

        Параметры:
            keyframes_df (pd.DataFrame): таблица с ключевыми кадрами
            frames_folder (str): путь к папке с кадрами
            video_path (str): путь к видеофайлу
            output_pdf (str): имя выходного файла PDF
        """
        if keyframes_df.empty:
            print("❌ Нет ключевых кадров для создания отчёта")
            return

        # --- Регистрация шрифта ---
        try:
            pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))
            font_name = 'DejaVu'
        except Exception as e:
            print(f"⚠️ Не удалось загрузить DejaVuSans.ttf: {e}, используем стандартный шрифт")
            font_name = 'Helvetica'

        doc = SimpleDocTemplate(output_pdf, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # --- Заголовок с названием видео ---
        video_title = os.path.basename(video_path)
        title = Paragraph(f"<b>Report video:</b> {video_title}", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 24))

        # --- Таблица с данными ---
        data = [["Number", "Time (s)", "Path frame"]]

        for _, row in keyframes_df.iterrows():
            frame_num = int(row['frame_num'])
            frame_path = os.path.join(frames_folder, f"frame_{frame_num:0{digits}d}.jpg")
            if os.path.exists(frame_path):
                data.append([frame_num, round(row['timestamp_sec'], 2), frame_path])
            else:
                print(f"⚠️ Файл {frame_path} не найден для отчёта")

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 36))

        # --- Превью кадров ---
        for _, row in keyframes_df.iterrows():
            frame_num = int(row['frame_num'])
            frame_path = os.path.join(frames_folder, f"frame_{frame_num:0{digits}d}.jpg")

            if not os.path.exists(frame_path):
                print(f"⚠️ Кадр {frame_path} не найден для отчёта")
                continue

            img = RLImage(frame_path, width=300, height=200)
            caption = Paragraph(f"Frame {frame_num} | {round(row['timestamp_sec'], 2)} sec", styles['Normal'])
            caption.fontName = font_name
            elements.append(img)
            elements.append(caption)
            elements.append(Spacer(1, 12))

        # --- Сохранение PDF ---
        try:
            doc.build(elements)
            print(f"📄 Отчёт сохранён как {output_pdf}")
        except Exception as e:
            print(f"❌ Не удалось создать PDF-отчёт: {e}")
