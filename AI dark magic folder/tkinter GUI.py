from tkinter import *
import cv2
from PIL import Image, ImageTk, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as _TFDepthwiseConv2D
import sys

# --------------- Freaky AI stuff i don't understand ------------------
# ——— Custom shim to swallow the extra 'groups' arg ———
class DepthwiseConv2D(_TFDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # drop any 'groups' kwarg if present
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

# confirm versions
print("TensorFlow:", sys.modules['tensorflow'].__version__)
print("Pillow:", Image.__version__)
print("NumPy:", np.__version__)

# disable oneDNN if you want bit-exact results
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ——— Load model with our custom layer class ———
model = load_model(
    "keras_Model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": DepthwiseConv2D}
)

# ——— Labels ———
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]


# ----------------------------Tkinter setup stuff--------------------------------

cam = cv2.VideoCapture(0)

window = Tk()
window.geometry("500x500")

label = Label(window)
label.pack()

text1 = Label(window)
text2 = Label(window)
text1.pack()
text2.pack()

cam_running = False

after_id = None

def open_camera():
    global cam_running, after_id
    cam_running = True

    # Delete knapperne fra tidligere menu
    button_camera.pack_forget()
    button_picture.pack_forget()
    button_quit.pack_forget()

    # Check om kameraet bør køre
    if cam_running == True:

        # Få cam på GUI
        ret, frame = cam.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo_image = ImageTk.PhotoImage(image=img)
        label.photo_image = photo_image
        label.configure(image=photo_image)

        # Image stuff til AI tingen
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        arr = np.asarray(img).astype(np.float32)
        arr = (arr / 127.5) - 1.0
        batch = np.expand_dims(arr, axis=0)

        # AI stuff idk
        pred = model.predict(batch, verbose=0)
        i = int(np.argmax(pred[0]))

        text1.config(text=(f"Class: {str(class_names[i])}"))
        text2.config(text=(f"Confidence: {100 * pred[0][i]:.4f}%"))

        after_id = label.after(10, open_camera)

        button_back.pack()

def png():
    # Delete knapperne fra tidligere menu
    button_camera.pack_forget()
    button_picture.pack_forget()
    button_quit.pack_forget()

    # Image stuff
    image_path = "img.png"
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)

    arr = np.asarray(img).astype(np.float32)
    arr = (arr / 127.5) - 1.0
    batch = np.expand_dims(arr, axis=0)

    # ——— Predict ———
    pred = model.predict(batch,verbose=0)
    i = int(np.argmax(pred[0]))
    text1.config(text=(f"Class: {str(class_names[i])}"))
    text2.config(text=(f"Confidence: {100 * pred[0][i]:.4f}%"))
    button_back.pack()

def main_menu():
    global cam_running, after_id
    cam_running = False

    # Stop camera fra at loop sig self recursively
    if after_id is not None:
        label.after_cancel(after_id)
        after_id = None

    # Fjern cam billedet fra skærmen
    label.config(image="")
    label.photo_image = None

    button_back.pack_forget()
    text1.pack_forget()
    text2.pack_forget()

    button_camera.pack()
    button_picture.pack()
    button_quit.pack()

button_camera = Button(window, text="Use Camera",width=50, command=open_camera)
button_picture = Button(window, text="Use picture",width=50, command=png)
button_quit = Button(window, text="Quit",width=50,command=quit)
button_back = Button(window, text="Return to menu",width=50,command=main_menu)

main_menu()

window.mainloop()
