import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

line_x = 300

in_count = 0
out_count = 0

track_positions = {}

# ✅ FULL SCREEN SETUP
cv2.namedWindow("People Counter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("People Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ❌ Removed resize for fullscreen

    # Tracking
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {int(track_id)}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Counting logic
            if track_id in track_positions:
                prev_x = track_positions[track_id]

                # LEFT → RIGHT (IN)
                if prev_x < line_x and cx >= line_x:
                    in_count += 1

                # RIGHT → LEFT (OUT)
                elif prev_x > line_x and cx <= line_x:
                    out_count += 1

            track_positions[track_id] = cx

    # Draw line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 3)

    # UI boxes (bottom)
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h-100), (w//2, h), (0, 0, 255), -1)
    cv2.rectangle(frame, (w//2, h-100), (w, h), (0, 255, 0), -1)

    cv2.putText(frame, f"OUT: {out_count}", (50, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

    cv2.putText(frame, f"IN: {in_count}", (w//2 + 50, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

    # Title bar
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 128, 0), -1)
    cv2.putText(frame, "PEOPLE COUNTING", (w//2 - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()