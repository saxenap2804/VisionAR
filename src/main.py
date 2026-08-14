import cv2


def main():
    """Start the VisionAR webcam feed."""

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to access the webcam.")

    print("VisionAR started.")
    print("Press Q to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Unable to read frame.")
            break

        cv2.putText(
            frame,
            "VisionAR",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("VisionAR - Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()