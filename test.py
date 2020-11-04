import cv2
import numpy as np
from keras.models import load_model

cap = cv2.VideoCapture(0)

class_name = ['giang', 'hoang']

my_model = load_model("modal.h5")

while (True):
    # Capture frame-by-frame
    ret, image_org = cap.read()
    if not ret:
        continue

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    # Hiển thị
    if (len(faces) > 0):
        (x, y, w, h) = faces[0]
        image = image_org[y:y + h, x:x + w]
        cv2.rectangle(image_org, (x, y), (x + w, y + h), (255, 0, 0), 2)
        image = cv2.resize(image, dsize=(128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image).reshape(128, 128, 1)
        # Convert to tensor
        image = np.expand_dims(image, axis=0)

        # Predict
        predict = my_model.predict(image)
        # print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
        print("This picture is: ", predict)
        print(np.max(predict[0], axis=0))
        if np.max(predict) >= 0.1:
            # Show image
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1.5
            color = (0, 255, 0)
            thickness = 2

            cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
