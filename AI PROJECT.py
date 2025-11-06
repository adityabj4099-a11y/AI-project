import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

# --------- Paths ---------
dataset_path = "faces"
trainer_path = "trainer.yml"
labels_path = "labels.npy"
attendance_folder = "attendance"

os.makedirs(dataset_path, exist_ok=True)
os.makedirs(attendance_folder, exist_ok=True)

# --------- GUI Setup ---------
root = tk.Tk()
root.title("Face Recognition Attendance")

tk.Label(root, text="UID").grid(row=0, column=0, padx=10, pady=5)
uid_entry = tk.Entry(root)
uid_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Name").grid(row=1, column=0, padx=10, pady=5)
name_entry = tk.Entry(root)
name_entry.grid(row=1, column=1, padx=10, pady=5)

status_label = tk.Label(root, text="", fg="green")
status_label.grid(row=2, column=0, columnspan=2, pady=5)

# --------- Step 1: Capture Faces ---------
def capture_faces():
    uid = uid_entry.get().strip()
    name = name_entry.get().strip()

    if uid == "" or name == "":
        messagebox.showwarning("Input Error", "Please enter both UID and Name!")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Cannot access webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            filename = f"{dataset_path}/{uid}_{name}_{count}.jpg"
            cv2.imwrite(filename, face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capturing Faces - Press 'Q' to Quit", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')] or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Capture Complete", f"Captured {count} images for {name} (UID: {uid})")
    status_label.config(text=f"✅ {count} images captured for {name} (UID: {uid})")

# --------- Step 2: Train Recognizer ---------
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_dict = {}
    current_id = 0

    for file in os.listdir(dataset_path):
        if file.endswith(".jpg"):
            path = os.path.join(dataset_path, file)
            gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            uid = file.split("_")[0]
            name = file.split("_")[1]
            if uid not in label_dict:
                label_dict[uid] = {"id": current_id, "name": name}
                current_id += 1
            faces.append(gray_img)
            labels.append(label_dict[uid]["id"])

    if not faces:
        messagebox.showwarning("No Data", "No faces found! Capture first.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(trainer_path)
    np.save(labels_path, label_dict)
    messagebox.showinfo("Training", "✅ Training completed successfully!")
    status_label.config(text="✅ Training completed!")

# --------- Step 3: Mark Attendance ---------
def mark_attendance():
    if not os.path.exists(trainer_path):
        messagebox.showwarning("Error", "Train recognizer first!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)

    label_dict = np.load(labels_path, allow_pickle=True).item()
    id_to_uid_name = {v["id"]: (uid, v["name"]) for uid, v in label_dict.items()}

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    attendance = {}

    messagebox.showinfo("Info", "Press 'Q' to stop attendance window")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            id_, conf = recognizer.predict(face_img)

            if conf < 55:
                uid, name = id_to_uid_name[id_]
                now = datetime.now().strftime("%H:%M:%S")
                attendance[uid] = {"Name": name, "Time": now}
                cv2.putText(frame, f"{name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Attendance - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

    if attendance:
        today = datetime.now().strftime("%Y-%m-%d")
        data = [
            {"UID": uid, "Name": info["Name"], "Time": info["Time"], "Date": today}
            for uid, info in attendance.items()
        ]
        df = pd.DataFrame(data)
        csv_file = os.path.join(attendance_folder, f"attendance_{today}.csv")
        df.to_csv(csv_file, index=False)
        messagebox.showinfo("Saved", f"✅ Attendance saved: {csv_file}")
        status_label.config(text=f"✅ Attendance saved for {len(attendance)} people")
    else:
        messagebox.showwarning("No Face Detected", "No attendance recorded!")

# --------- Buttons ---------
tk.Button(root, text="Capture Faces", command=capture_faces).grid(row=3, column=0, columnspan=2, pady=5)
tk.Button(root, text="Train Recognizer", command=train_recognizer).grid(row=4, column=0, columnspan=2, pady=5)
tk.Button(root, text="Mark Attendance", command=mark_attendance).grid(row=5, column=0, columnspan=2, pady=5)

root.mainloop()
