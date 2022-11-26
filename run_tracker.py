import os
import cv2

from tracker import load_detector, load_tracker, detect


def run():
    model, stride, names, pt, device = load_detector(weights=os.path.join("weights", "yolov5m.pt"))
    tracker_list, outputs = load_tracker(device=device)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputs = detect(model, names, frame, tracker_list, outputs, device=device, draw_detections=True)
        for output in outputs:
            print(f"Tracks: {output}")

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
