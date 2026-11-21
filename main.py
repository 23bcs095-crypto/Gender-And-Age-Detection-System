
"""Detect faces and predict age range and gender using ViT."""

import argparse
from pathlib import Path

import cv2

from age_gender_model import predict_age_gender


BASE_DIR = Path(__file__).resolve().parent

FACE_PROTO = BASE_DIR / "opencv_face_detector.pbtxt"
FACE_MODEL = BASE_DIR / "opencv_face_detector_uint8.pb"

WINDOW_NAME = "Age and Gender Detection"


def load_face_detector():
    """Load OpenCV face detector."""

    required = (FACE_PROTO, FACE_MODEL)

    missing = [
        path.name for path in required
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing face detector files: " + ", ".join(missing)
        )

    return cv2.dnn.readNet(
        str(FACE_MODEL),
        str(FACE_PROTO)
    )


def detect_and_annotate(frame, face_net):
    """Detect faces and display age range and gender."""

    height, width = frame.shape[:2]
    display = frame.copy()

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        (104, 117, 123),
        swapRB=True,
        crop=False
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    found_faces = 0

    for index in range(detections.shape[2]):

        face_confidence = float(
            detections[0, 0, index, 2]
        )

        if face_confidence < 0.7:
            continue

        x1, y1, x2, y2 = (
            int(
                detections[0, 0, index, coordinate]
                * dimension
            )
            for coordinate, dimension in zip(
                (3, 4, 5, 6),
                (width, height, width, height)
            )
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        padding = 20

        face = frame[
            max(0, y1 - padding):min(height, y2 + padding),
            max(0, x1 - padding):min(width, x2 + padding)
        ]

        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        prediction = predict_age_gender(face_rgb)

        age_range = (
            prediction["age_range"]
            .replace("–", "-")
            .replace(" years", "")
        )

        gender = prediction["gender"]
        gender_confidence = prediction["gender_confidence"]

        gender_label = f"{gender} {gender_confidence:.0%}"
        age_label = age_range

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        gender_size = cv2.getTextSize(
            gender_label, font, font_scale, thickness
        )[0]

        age_size = cv2.getTextSize(
            age_label, font, font_scale, thickness
        )[0]

        text_width = max(
            gender_size[0],
            age_size[0]
        )

        text_x = min(
            max(5, x1),
            max(5, width - text_width - 10)
        )

        if y1 >= 55:
            gender_y = y1 - 30
            age_y = y1 - 5
        else:
            gender_y = min(height - 40, y2 + 30)
            age_y = min(height - 10, y2 + 55)

        cv2.putText(
            display,
            gender_label,
            (text_x, gender_y),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            age_label,
            (text_x, age_y),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        print(f"{gender_label}, Age: {age_label}")

        found_faces += 1

    return display, found_faces


def open_fullscreen_window():
    """Create a fullscreen OpenCV window."""

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--image",
        help="Image file to analyze; omit to use webcam."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0)."
    )

    parser.add_argument(
        "--output",
        help="Save annotated image to this path."
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a preview window."
    )

    args = parser.parse_args()

    face_net = load_face_detector()

    # Image mode
    if args.image:

        image_path = Path(args.image).expanduser()
        frame = cv2.imread(str(image_path))

        if frame is None:
            parser.error(
                f"Could not read image: {image_path}"
            )

        result, count = detect_and_annotate(
            frame,
            face_net
        )

        print(f"Faces detected: {count}")

        if args.output:
            if not cv2.imwrite(args.output, result):
                parser.error(
                    f"Could not save output image: {args.output}"
                )

            print(f"Output saved: {args.output}")

        if not args.no_display:

            open_fullscreen_window()

            cv2.imshow(WINDOW_NAME, result)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return

    # Webcam mode
    if args.output:
        parser.error(
            "--output can only be used with --image."
        )

    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        parser.error(
            f"Could not open camera {args.camera}."
        )

    open_fullscreen_window()

    print("Webcam started in fullscreen mode. Press q to quit.")

    try:

        while True:

            has_frame, frame = camera.read()

            if not has_frame:
                print("Could not read webcam frame.")
                break

            result, _ = detect_and_annotate(
                frame,
                face_net
            )

            cv2.imshow(
                WINDOW_NAME,
                result
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
