import os
import cv2

label = "hoang"

cap = cv2.VideoCapture(0)

i = 0
while (True):
    ret, frame = cap.read()
    if not ret:
        continue
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    # Hiển thị
    if (len(faces) > 0):
        i += 1
        (x, y, w, h) = faces[0]
        img = frame[y:y + h, x:x + w]
        cv2.imshow('frame', img)
        # Lưu dữ liệu
        if i >= 60 and i <= 2060:
            print("Số ảnh capture = ", i - 60)
            # Tạo thư mục nếu chưa có
            if not os.path.exists('data/' + str(label)):
                os.mkdir('data/' + str(label))

            cv2.imwrite('data/' + str(label) + "/" + str(i) + ".png", img)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
