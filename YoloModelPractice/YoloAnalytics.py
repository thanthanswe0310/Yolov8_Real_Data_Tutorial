import cv2

from ultralytics import YOLO,solutions

model = YOLO("/home/tts/Desktop/TTS_CV/YoloModelPractice/best.pt")

cap = cv2.VideoCapture("/home/tts/Desktop/TTS_CV/EJ_Hand_Tracking/generated_videos/banana_uyu_left.mp4")


assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1920, 1080),  # This is fixed
)

analytics = solutions.Analytics(
    analytics_type="line",
    show=True,
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        im0 = analytics.process_data(im0, frame_count)  # update analytics graph every frame
        out.write(im0)  # write the video file
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()