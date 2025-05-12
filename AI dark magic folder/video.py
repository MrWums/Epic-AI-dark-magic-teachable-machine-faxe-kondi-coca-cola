import sys
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as _TFDepthwiseConv2D
from tensorflow.keras import backend as K
from PIL import Image, ImageOps
import numpy as np
import cv2

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

# ——— Image preprocessing ———
image_path = "img.png"
img = Image.open(image_path).convert("RGB")
img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    # Display the captured frame
    cv2.imshow('Camera', frame)


    # AI dark magic
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    arr = (arr / 127.5) - 1.0
    batch = np.expand_dims(arr, axis=0)

    # ——— Predict ———
    pred = model.predict(batch)
    i = int(np.argmax(pred[0]))
    print(f"Class: {class_names[i]}")
    print(f"Confidence: {pred[0][i]:.4f}")

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break