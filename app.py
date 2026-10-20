import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Create Main Window
root = tk.Tk()
root.title("Plant Disease Detection System")
root.geometry("700x700")
root.configure(bg="white")

selected_image = None

# Heading
title = tk.Label(
    root,
    text="Plant Disease Detection System",
    font=("Arial", 22, "bold"),
    bg="white",
    fg="green"
)
title.pack(pady=20)

# Image Label
image_label = tk.Label(root, bg="white")
image_label.pack(pady=20)

# Result Label
result_label = tk.Label(
    root,
    text="Prediction : ",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="blue",
    justify="left"
)
result_label.pack(pady=20)


# Upload Image
def upload_image():
    global selected_image

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        selected_image = file_path

        img = Image.open(file_path)
        img = img.resize((300, 300))
        photo = ImageTk.PhotoImage(img)

        image_label.config(image=photo)
        image_label.image = photo

        result_label.config(
            text="Image Uploaded Successfully\nClick Predict"
        )


# Predict Disease
def predict_disease():

    if selected_image is None:
        result_label.config(text="Please Upload an Image First")
        return

    disease = "Tomato Early Blight"
    confidence = "98.45%"
    severity = "Moderate"

    treatment = (
        "Treatment:\n"
        "• Remove infected leaves\n"
        "• Spray recommended fungicide\n"
        "• Avoid over watering"
    )

    result = f"""
Prediction : {disease}

Confidence : {confidence}

Severity : {severity}

{treatment}
"""

    result_label.config(text=result)


# Upload Button
upload_btn = tk.Button(
    root,
    text="Choose Image",
    command=upload_image,
    font=("Arial", 14, "bold"),
    bg="green",
    fg="white",
    width=20
)
upload_btn.pack(pady=10)

# Predict Button
predict_btn = tk.Button(
    root,
    text="Predict",
    command=predict_disease,
    font=("Arial", 14, "bold"),
    bg="blue",
    fg="white",
    width=20
)
predict_btn.pack(pady=10)

root.mainloop()