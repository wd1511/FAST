import os
import cv2

name = 'b6'
video_path = './video/{}.mp4'.format(name)

frame_indices = [0, 4, 8, 12, 16, 20]

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

extracted_frames = []

for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    if i in frame_indices:
        extracted_frames.append(frame)
        frame_indices.remove(i) 
    if not frame_indices:
        break
cap.release()

for idx, frame in enumerate(extracted_frames):
    output_path = f'./image/{name}_{idx}.jpg'
    cv2.imwrite(output_path, frame)
    print(f'Frame saved to {output_path}')