#!/usr/bin/env python3
"""Split scanned sheets with multiple photos into separate image files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import dlib
except ImportError:  # pragma: no cover - optional dependency
    dlib = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Найти отдельные фотографии на сканированном изображении, "
            "выпрямить их и сохранить в отдельные файлы."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Путь к сканированному изображению (JPEG/PNG и т.п.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Каталог для сохранения результатов (по умолчанию рядом с исходником)",
    )
    parser.add_argument(
        "--suffix",
        default="_part",
        help="Суффикс, добавляемый к исходному имени файла (по умолчанию '_part')",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.05,
        help=(
            "Минимальная доля площади снимка, которую должен занимать контур, "
            "чтобы считаться фотографией (по умолчанию 0.05)"
        ),
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        help="Размер ядра гауссова размытия перед поиском контуров (по умолчанию 5)",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=11,
        help=(
            "Размер прямоугольного ядра морфологического замыкания Canny-маски "
            "(по умолчанию 11). Увеличьте, если фотографий не видно полностью."
        ),
    )
    parser.add_argument(
        "--canny-thresholds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(50.0, 150.0),
        help="Пороговые значения для детектора Canny (по умолчанию 50 150)",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=0,
        help="Количество пикселов, которое нужно обрезать с каждой стороны готового фото",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Сохранить промежуточные изображения в указанную папку",
    )
    parser.add_argument(
        "--auto-orient",
        action="store_true",
        help="Автоматически определить ориентацию фотографии через dlib и повернуть её.",
    )
    parser.add_argument(
        "--face-upsamples",
        type=int,
        default=1,
        help="Количество апсемплирований для детектора лиц dlib (по умолчанию 1)",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return consistent ordering of four corner points: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def max_side_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def warp_photo(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    ordered = order_points(box)

    (tl, tr, br, bl) = ordered
    width_top = max_side_length(tr, tl)
    width_bottom = max_side_length(br, bl)
    max_width = int(round(max(width_top, width_bottom)))

    height_left = max_side_length(bl, tl)
    height_right = max_side_length(br, tr)
    max_height = int(round(max(height_left, height_right)))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def preprocess(
    image: np.ndarray,
    blur_kernel: int,
    close_kernel: int,
    canny_low: float,
    canny_high: float,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur_kernel > 1:
        blur_kernel = blur_kernel + (1 - blur_kernel % 2)
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    edges = cv2.Canny(gray, canny_low, canny_high)

    close_kernel = max(close_kernel, 3)
    if close_kernel % 2 == 0:
        close_kernel += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, None, iterations=1)
    return closed


def find_photo_contours(mask: np.ndarray, min_area: float) -> list[tuple[np.ndarray, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: list[tuple[np.ndarray, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if min(width, height) < 20:
            continue
        box = cv2.boxPoints(rect)
        result.append((box, area))
    return result


def crop_margin(image: np.ndarray, crop: int) -> np.ndarray:
    crop = max(int(crop), 0)
    if crop == 0:
        return image
    height, width = image.shape[:2]
    x0 = min(crop, width // 2)
    y0 = min(crop, height // 2)
    x1 = width - x0
    y1 = height - y0
    if x1 <= x0 or y1 <= y0:
        return image
    return image[y0:y1, x0:x1]


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle % 360 == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270 or angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation angle: {angle}")


def resize_for_analysis(image: np.ndarray, max_side: int = 800) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


class DlibFaceOrientation:
    def __init__(self, upsample: int = 1) -> None:
        if dlib is None:
            raise RuntimeError("dlib is not available")
        self.detector = dlib.get_frontal_face_detector()
        self.upsample = max(int(upsample), 0)

    def _detect_faces(self, image: np.ndarray) -> int:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            rectangles, scores, _ = self.detector.run(rgb, self.upsample, 0)
            count = len(rectangles)
        except Exception:  # pragma: no cover - fallback path for older dlib
            rectangles = self.detector(rgb, self.upsample)
            count = len(rectangles)
        return int(count)

    def estimate(self, image: np.ndarray) -> int:
        sampled = resize_for_analysis(image)
        scores: dict[int, int] = {}
        for angle in (0, 90, 180, 270):
            rotated = rotate_image(sampled, angle)
            scores[angle] = self._detect_faces(rotated)

        best_angle = max(scores, key=scores.get)
        best_score = scores[best_angle]
        if best_score <= 0 or best_angle == 0:
            return 0

        # Require unique best rotation; ties mean недостаточно уверенности.
        second_best = max((score for angle, score in scores.items() if angle != best_angle), default=0)
        if best_score == second_best:
            return 0

        return best_angle


def auto_orient_photo(
    image: np.ndarray, estimator: DlibFaceOrientation | None
) -> tuple[np.ndarray, int]:
    if estimator is None:
        return image, 0
    rotation = estimator.estimate(image)
    if rotation == 0:
        return image, 0
    return rotate_image(image, rotation), rotation


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Файл {args.input} не найден", file=sys.stderr)
        return 1

    image = cv2.imread(str(args.input))
    if image is None:
        print(
            "Не удалось прочитать изображение. Поддерживаются форматы, которые умеет OpenCV.",
            file=sys.stderr,
        )
        return 1

    output_dir = args.output_dir or args.input.parent
    ensure_dir(output_dir)
    ensure_dir(args.debug_dir)

    orientation_estimator: DlibFaceOrientation | None = None
    if args.auto_orient:
        if dlib is None:
            print(
                "Автоориентация выключена: модуль dlib не установлен.",
                file=sys.stderr,
            )
        else:
            try:
                orientation_estimator = DlibFaceOrientation(args.face_upsamples)
            except Exception as exc:  # pragma: no cover - initialization errors
                print(
                    f"Автоориентация выключена: не удалось инициализировать dlib ({exc}).",
                    file=sys.stderr,
                )
                orientation_estimator = None

    mask = preprocess(
        image,
        args.blur_kernel,
        args.close_kernel,
        *args.canny_thresholds,
    )

    total_area = float(image.shape[0] * image.shape[1])
    min_area = total_area * max(args.min_area_ratio, 0.0)
    photo_contours = find_photo_contours(mask, min_area)

    if not photo_contours:
        print(
            "Фотографии не найдены. Попробуйте уменьшить --min-area-ratio или настроить пороги.",
            file=sys.stderr,
        )
        return 2

    def contour_key(item: tuple[np.ndarray, float]) -> tuple[float, float]:
        box, _ = item
        center = box.mean(axis=0)
        return (center[1], center[0])

    photo_contours.sort(key=contour_key)

    base_name = args.input.stem
    extension = args.input.suffix or ".jpg"
    crop = max(args.crop, 0)

    for index, (box, _area) in enumerate(photo_contours, start=1):
        warped = warp_photo(image, box.astype("float32"))
        if crop > 0:
            warped = crop_margin(warped, crop)
        if args.auto_orient and orientation_estimator is not None:
            warped, rotation = auto_orient_photo(warped, orientation_estimator)
            if rotation:
                print(f"Фото {index} повернуто на {rotation} градусов")
        output_path = output_dir / f"{base_name}{args.suffix}{index:02d}{extension}"
        success = cv2.imwrite(str(output_path), warped)
        if not success:
            print(f"Не удалось сохранить {output_path}", file=sys.stderr)

    if args.debug_dir:
        debug_canvas = image.copy()
        for idx, (box, _) in enumerate(photo_contours, start=1):
            box_int = np.intp(box)
            cv2.drawContours(debug_canvas, [box_int], -1, (0, 255, 0), 3)
            center = box.mean(axis=0)
            cv2.putText(
                debug_canvas,
                str(idx),
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(args.debug_dir / f"{base_name}_debug{extension}"), debug_canvas)
        cv2.imwrite(str(args.debug_dir / f"{base_name}_mask.png"), mask)

    print(f"Найдено фотографий: {len(photo_contours)}")
    print(f"Результаты сохранены в {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
