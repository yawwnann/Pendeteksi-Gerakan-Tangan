import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Variabel untuk menyimpan posisi terakhir dari ujung jari
prev_x, prev_y = None, None

# Fungsi untuk menghitung jumlah jari yang terangkat
def count_fingers(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    count = 0
    
    # Ibu jari
    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x:
        count += 1
    
    # Jari telunjuk sampai kelingking
    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    
    return count

# Fungsi untuk mengkonversi jumlah jari ke teks
def fingers_to_text(finger_count):
    if finger_count == 1:
        return "satu"
    elif finger_count == 2:
        return "dua"
    elif finger_count == 3:
        return "tiga"
    elif finger_count == 4:
        return "empat"
    elif finger_count == 5:
        return "dada monyet"
    else:
        return ""

# Buka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Inisialisasi canvas untuk menggambar
canvas = None

while True:
    # Ambil frame dari kamera
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip gambar secara horizontal (mirror)
    image = cv2.flip(image, 1)

    # Buat canvas pada frame pertama
    if canvas is None:
        canvas = image.copy()
    else:
        # Salin canvas agar tidak menghapus gambar lama
        canvas[:] = canvas[:]

    # Konversi gambar ke format RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # Jika tangan terdeteksi
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Menggambar landmark dan koneksi pada gambar
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Hitung jumlah jari yang terangkat
            finger_count = count_fingers(hand_landmarks)
            
            # Konversi jumlah jari ke teks
            text = fingers_to_text(finger_count)
            
            # Tampilkan teks pada gambar
            cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Menggambar lingkaran di ujung jari yang terangkat
            finger_tips = [4, 8, 12, 16, 20]
            for tip in finger_tips:
                x = int(hand_landmarks.landmark[tip].x * image.shape[1])
                y = int(hand_landmarks.landmark[tip].y * image.shape[0])
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)  # Gambar lingkaran hijau

            # Gambar jika satu jari terangkat
            if finger_count == 1:
                x = int(hand_landmarks.landmark[8].x * image.shape[1])
                y = int(hand_landmarks.landmark[8].y * image.shape[0])

                if prev_x is None or prev_y is None:
                    prev_x, prev_y = x, y

                # Gambar garis dari posisi sebelumnya ke posisi sekarang pada canvas
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    # Tampilkan gambar asli dan canvas
    # Gunakan gambar asli untuk tampilan dan canvas untuk menggambar
    combined_image = cv2.addWeighted(image, 1, canvas, 0.5, 0)
    cv2.imshow('Hand Tracking', combined_image)

    # Tekan 'Esc' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Melepaskan resource
cap.release()
cv2.destroyAllWindows()
