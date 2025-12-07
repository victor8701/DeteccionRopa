from ultralytics import YOLO

# --- 1. PEGAR EL CÓDIGO DE ROBOFLOW ---
from roboflow import Roboflow
rf = Roboflow(api_key="6rvt97oIRw6HjjI3MQhp")
project = rf.workspace("foldedclothdetection").project("my-first-project-iepg5")
version = project.version(1)
dataset = version.download("yolov8")

# --- 2. CARGAMOS EL CEREBRO DE YOLO ---
# Usamos 'yolov8n.pt' (nano) que es el más rápido y ligero
model = YOLO('yolov8n.pt')

# --- 3. ENTRENAMOS CON TUS DATOS ---
# Fíjate en el truco: usamos 'dataset.location' para que encuentre tus fotos solo
print("Comenzando entrenamiento...")

model.train(
    data=f"{dataset.location}/data.yaml",  # La ruta automática a tus datos
    epochs=100,      # 100 épocas como acordamos para tus 14 clases
    imgsz=640,       # Tamaño de imagen estándar
    plots=True       # Para que te guarde gráficas de cómo aprende
)

print("¡Entrenamiento finalizado!")