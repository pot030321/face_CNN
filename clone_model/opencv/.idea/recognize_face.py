import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Tải mô hình đã huấn luyện
model = load_model('models/face_recognition_model.h5')
le = np.load('models/label_encoder.npy', allow_pickle=True)

# Hàm nhận diện khuôn mặt từ webcam
def recognize_face_from_webcam():
    cap = cv2.VideoCapture(0)  # Mở webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hiển thị video
        cv2.imshow("Press Space to capture", frame)

        # Xử lý ảnh và dự đoán nhãn
        img_resized = cv2.resize(frame, (64, 64))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Dự đoán nhãn
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        label = le[predicted_class]

        print(f"Predicted label: {label}")

        # Nhấn Space để dừng nhận diện
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print("Escape hit, closing...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm nhận diện khuôn mặt từ video
def recognize_face_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hiển thị video
        cv2.imshow("Press Space to capture", frame)

        # Xử lý ảnh và dự đoán nhãn
        img_resized = cv2.resize(frame, (64, 64))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Dự đoán nhãn
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        label = le[predicted_class]

        print(f"Predicted label: {label}")

        # Nhấn Space để dừng nhận diện
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print("Escape hit, closing...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function để người dùng chọn giữa webcam hoặc video
def main():
    print("Chọn phương thức nhận diện:")
    print("1. Nhận diện khuôn mặt qua webcam")
    print("2. Nhận diện khuôn mặt từ video")
    
    # Yêu cầu người dùng nhập lựa chọn
    choice = input("Nhập lựa chọn của bạn (1/2): ")

    if choice == '1':
        print("Đang nhận diện khuôn mặt qua webcam...")
        recognize_face_from_webcam()
    elif choice == '2':
        video_path = input("Nhập đường dẫn đến video: ")
        print(f"Đang nhận diện khuôn mặt từ video: {video_path}")
        recognize_face_from_video(video_path)
    else:
        print("Lựa chọn không hợp lệ!")

# Chạy chương trình chính
if __name__ == "__main__":
    main()
