import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Cấu hình thư mục chứa dữ liệu
data_dir = "data"  # Đảm bảo rằng thư mục "data" chứa các thư mục con (ví dụ, "person_1", "person_2", ...)

# Đọc dữ liệu từ thư mục
def load_data(data_dir):
    data = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        
        if os.path.isdir(label_dir):  # Kiểm tra nếu là thư mục con (mỗi thư mục con là một nhãn)
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                img = cv2.imread(image_path)
                img_resized = cv2.resize(img, (64, 64))  # Resize ảnh về kích thước cố định
                data.append(img_resized)
                labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)

    # Mã hóa nhãn
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    labels = to_categorical(labels)

    return data, labels, le

# Huấn luyện mô hình
def train_model():
    # Tải dữ liệu từ thư mục
    data, labels, le = load_data(data_dir)

    # Kiểm tra số lượng dữ liệu
    print(f"Number of training images: {len(data)}")

    # Kiểm tra nếu chỉ có một mẫu
    if len(data) == 1:
        print("Warning: Only one sample found. Training will proceed with this single sample.")

    # Xây dựng mô hình CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Áp dụng data augmentation
    datagen.fit(data)

    # Nếu chỉ có một mẫu, huấn luyện trực tiếp trên toàn bộ dữ liệu
    if len(data) == 1:
        history = model.fit(datagen.flow(data, labels, batch_size=1), epochs=50, verbose=1)
    else:
        # Chia dữ liệu thành train và validation nếu có nhiều mẫu
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
        history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                            epochs=50,
                            verbose=1,
                            validation_data=(X_val, y_val),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

    # Kiểm tra lịch sử huấn luyện
    print(f"Training history: {history.history}")

    # Lưu mô hình
    model.save('models/face_recognition_model.h5')
    print("Model training complete!")

    # Lưu label encoder
    np.save('models/label_encoder.npy', le.classes_)

# Chạy huấn luyện mô hình
train_model()
