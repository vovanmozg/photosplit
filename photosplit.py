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


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}


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
        help="Путь к папке с изображениями для обработки",
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
        gray_blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    else:
        gray_blurred = gray

    edges = cv2.Canny(gray_blurred, canny_low, canny_high)

    close_kernel = max(close_kernel, 3)
    if close_kernel % 2 == 0:
        close_kernel += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    edge_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    _, thresh = cv2.threshold(
        gray_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(close_kernel // 2, 3),) * 2)
    filled = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, fill_kernel, iterations=2)

    combined = cv2.bitwise_or(edge_mask, filled)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY)
    return binary


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


def process_image_file(
    image_path: Path,
    args: argparse.Namespace,
    orientation_estimator: DlibFaceOrientation | None,
) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        print(
            f"[{image_path.name}] Пропущено: не удалось прочитать изображение.",
            file=sys.stderr,
        )
        return False

    output_dir = args.output_dir or image_path.parent
    ensure_dir(output_dir)

    debug_dir = None
    if args.debug_dir:
        debug_dir = args.debug_dir / image_path.stem
        ensure_dir(debug_dir)

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
            f"[{image_path.name}] Фотографии не найдены."
            " Попробуйте настроить параметры обнаружения.",
            file=sys.stderr,
        )
        return False

    def contour_key(item: tuple[np.ndarray, float]) -> tuple[float, float]:
        box, _ = item
        center = box.mean(axis=0)
        return (center[1], center[0])

    photo_contours.sort(key=contour_key)

    base_name = image_path.stem
    extension = image_path.suffix or ".jpg"
    crop = max(args.crop, 0)
    processed = 0

    for index, (box, _area) in enumerate(photo_contours, start=1):
        warped = warp_photo(image, box.astype("float32"))
        if crop > 0:
            warped = crop_margin(warped, crop)
        if orientation_estimator is not None:
            warped, rotation = auto_orient_photo(warped, orientation_estimator)
            if rotation:
                print(
                    f"[{image_path.name}] Фото {index} повернуто на {rotation} градусов"
                )
        output_path = output_dir / f"{base_name}{args.suffix}{index:02d}{extension}"
        success = cv2.imwrite(str(output_path), warped)
        if not success:
            print(f"[{image_path.name}] Не удалось сохранить {output_path}", file=sys.stderr)
        else:
            processed += 1

    if debug_dir is not None:
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
        cv2.imwrite(
            str(debug_dir / f"{base_name}_debug{extension}"),
            debug_canvas,
        )
        cv2.imwrite(str(debug_dir / f"{base_name}_mask.png"), mask)

    print(f"[{image_path.name}] Найдено фотографий: {len(photo_contours)}")
    print(f"[{image_path.name}] Сохранено: {processed}, каталог: {output_dir}")
    return processed > 0


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Каталог {args.input} не найден", file=sys.stderr)
        return 1

    if not args.input.is_dir():
        print(f"{args.input} не является каталогом", file=sys.stderr)
        return 1

    if args.debug_dir:
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

    image_paths = sorted(
        path
        for path in args.input.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print("В каталоге нет поддерживаемых изображений.", file=sys.stderr)
        return 2

    successes = 0
    for image_path in image_paths:
        try:
            if process_image_file(image_path, args, orientation_estimator):
                successes += 1
        except Exception as exc:  # pragma: no cover - safeguards
            print(f"[{image_path.name}] Ошибка обработки: {exc}", file=sys.stderr)

    processed_total = len(image_paths)
    print(
        f"Готово: обработано {successes} из {processed_total} файлов"
        f" (каталог {args.input})."
    )
    return 0 if successes else 3


if __name__ == "__main__":
    raise SystemExit(main())
