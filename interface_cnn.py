import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Liste des classes
labels_names = ['Avion', 'Voiture', 'Oiseau', 'Chat', 'Cerf', 'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion']

# 🔮 Fonction prédiction
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return labels_names[predicted_class]

# 🖼️ Interface Tkinter
def run_interface():
    # Charger modèle
    model_path = filedialog.askopenfilename(title="Sélectionner le fichier .h5 du modèle", filetypes=[("H5 files", "*.h5")])
    if not model_path:
        print("Modèle non sélectionné.")
        return
    model = load_model(model_path)

    def load_and_predict():
        file_path = filedialog.askopenfilename(title="Choisir une image")
        if file_path:
            img = Image.open(file_path).resize((150, 150))
            tk_img = ImageTk.PhotoImage(img)
            image_label.config(image=tk_img)
            image_label.image = tk_img

            predicted_label = predict_image(model, file_path)
            result_label.config(text=f"✅ Prédiction : {predicted_label}")

    # Création de la fenêtre
    root = tk.Tk()
    root.title("Reconnaissance d'objet - CIFAR-10")
    root.geometry("400x450")

    btn = tk.Button(root, text="📂 Choisir une image", command=load_and_predict)
    btn.pack(pady=10)

    image_label = tk.Label(root)
    image_label.pack()

    result_label = tk.Label(root, text="Résultat : ", font=("Arial", 14))
    result_label.pack(pady=20)

    root.mainloop()

# 🟢 Exécuter l'interface
run_interface()
