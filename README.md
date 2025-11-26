# Week-4-Assignment-AI-in-Software-Engineering 

This assignment evaluates your understanding of emerging AI trends, their technical implementations, and ethical implications. Through theoretical analysis, hands-on projects, and critical reflection, you will explore Edge AI, AI-IoT integration, Human-AI collaboration, Quantum AI, Personalized Medicine, and Ethical Challenges.



Submission Guidelines

Code/Diagrams: Well-documented scripts/Readme File, Jupyter notebooks. (Submitted on GitHub)



Tools & Resources

Frameworks: TensorFlow Lite, any other of your choice.

Datasets: Kaggle (Edge AI), TCGA (Personalized Medicine). Any of your choice
Part 1: Theoretical Analysis
1. Essay Questions

Q1: Explain how Edge AI reduces latency and enhances privacy compared to cloud-based AI. Provide a real-world example (e.g., autonomous drones).

Q2: Compare Quantum AI and classical AI in solving optimization problems. What industries could benefit most from Quantum AI?

Part 2: Practical Implementation
Task 1: Edge AI Prototype

Tools: TensorFlow Lite, Raspberry Pi/Colab (simulation).

Goal:

Train a lightweight image classification model (e.g., recognizing recyclable items).

Convert the model to TensorFlow Lite and test it on a sample dataset.

Explain how Edge AI benefits real-time applications.

Deliverable: Code + report with accuracy metrics and deployment steps.

Task 2: AI-Driven IoT Concept

Scenario: Design a smart agriculture simulation system using AI and IoT.

Requirements:

List sensors needed (e.g., soil moisture, temperature).

Propose an AI model to predict crop yields.

Sketch a data flow diagram (AI processing sensor data).



Tips: Be free to use any tools of your choice
ChatGPT said:

Thought for 5s
Nice — I’ll build a complete, ready-to-copy assignment package you can submit to GitHub: short, clear theoretical answers, a practical Edge-AI prototype (code + TFLite conversion + evaluation), a smart-agriculture AI-IoT design (sensors, model, data-flow sketch), and a README / deliverable checklist. I’ll keep code runnable in Colab or a local machine and the deployment steps targeted at Raspberry Pi. Here we go.

AI Future Directions — Submission Package
Part 1 — Theoretical Analysis
Q1 — How Edge AI reduces latency and enhances privacy (with example)
Short answer (core ideas).
Edge AI runs inference (and optionally training/updates) on devices at or near the data source rather than sending raw data to the cloud. That reduces round-trip time (lowers latency) because no network round-trip is required for each inference. It enhances privacy because sensitive raw data (images, audio, personal sensor readings) can be processed locally and only anonymized summaries, model updates, or results are sent to the cloud (or not sent at all).

Mechanisms that reduce latency

Local inference avoids network RTT and congestion.

Lightweight models (quantized, pruned) run faster on small CPUs/NPUs.

Real-time scheduling on-device reduces jitter and variability.

Mechanisms that enhance privacy

Raw data never leaves device (on-device processing).

Differential privacy or secure aggregation can be applied at the edge before any sharing.

Federated learning: model updates (encrypted/gradient-level) can be shared rather than raw data.

Real-world example — autonomous drones (edge AI)

Scenario: surveillance or search-and-rescue drone that must detect people, obstacles, or signs of distress.

On-drone inference allows immediate obstacle avoidance and person detection with <100 ms reaction time (no cloud RTT).

Privacy: raw video is analyzed on-board; only bounding boxes, metadata, or anonymized alerts are transmitted.

Bandwidth savings: only critical events are sent, not continuous video streams.

Safety: if connectivity drops, drone still can operate safely using local models.

Tradeoffs & considerations

Edge devices have limited compute/energy — models must be optimized.

Model updates and global consistency require strategies (federated learning, periodic cloud sync).

Security of device (tamper protection) and model integrity are important.

Q2 — Quantum AI vs Classical AI in optimization problems
What “Quantum AI” refers to (brief).
Quantum AI usually means using quantum computing algorithms (quantum annealing, gate-model algorithms like QAOA) to solve optimization/search problems that are hard for classical methods, possibly integrated with classical ML (hybrid algorithms).

Comparison on optimization problems

Aspect	Classical AI / classical optimization	Quantum approaches
Algorithm maturity	Very mature (gradient methods, simulated annealing, branch & bound, metaheuristics)	Emerging; noisy hardware and hybrid methods common
Scalability	Scales well on many problems; can exploit heuristics and approximate solutions	Potential exponential speedups on specific problem classes (unproven for all real-world cases); limited by qubit count and noise
Solution quality	Often finds near-optimal solutions reliably with heuristics	For certain problems (e.g., some combinatorial problems), quantum annealing or QAOA may find better/ faster solutions — research ongoing
Hardware constraints	Commodity hardware available	Few qubits, error rates, need for error mitigation
When beneficial	Problems with continuous differentiable objectives, large datasets	Combinatorial optimization, certain NP-hard problems, sampling, and problems benefiting from quantum superposition/tunneling
Quantum methods likely to help most

Combinatorial optimization: vehicle routing, scheduling, portfolio optimization.

Large-scale sampling problems: probabilistic modeling, Bayesian inference acceleration.

Materials discovery and molecular simulation (quantum chemistry).

Logistics and supply-chain optimization where combinatorial explosion occurs.

Industries that could benefit most

Logistics & transportation (routing, scheduling).

Finance (portfolio optimization, risk modeling).

Drug discovery and materials science (quantum chemistry workloads).

Manufacturing/supply chain (complex scheduling, resource allocation).

Energy (grid optimization, efficient dispatch).

Caveats

Quantum advantage is problem-specific and not yet general-purpose.

Hybrid classical-quantum pipelines (classical pre/post-processing with quantum core) are the pragmatic near-term path.

Part 2 — Practical Implementation
Below I give a practical Edge-AI prototype you can run in Colab or adapt for Raspberry Pi. It includes training a small image classifier (e.g., recyclable items) using transfer learning (MobileNetV2), converting to TFLite with quantization, test/evaluate, and deployment steps.

Task 1 — Edge AI Prototype: Recognizing recyclable items (code + notes)
Approach overview
Use transfer learning (MobileNetV2) to train a small classifier on a directory-structured dataset: data/train/<class>, data/val/<class>. Classes: plastic, glass, metal, paper, other.

Fine-tune a small head on top of MobileNetV2 (feature extractor).

Export saved model → convert to TensorFlow Lite with quantization (float16 or int8) for Raspberry Pi.

Evaluate test accuracy; produce TFLite file ready for on-device inference.

Colab / Jupyter notebook code
(You can paste this into a Colab notebook. Assumes TensorFlow 2.x.)

# Colab-ready notebook snippet
# Run this cell to install dependencies (uncomment in Colab)
# !pip install -q tensorflow==2.12 tensorflow-intel

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ====== User variables ======
DATA_DIR = "/content/data"   # point to your dataset in Colab (upload or mount Drive)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 8
MODEL_SAVE_PATH = "/content/recycle_classifier"
TFLITE_FILENAME = "/content/recycle_classifier.tflite"
# ============================

# Prepare datasets using Keras preprocessing
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Build model (MobileNetV2 as feature extractor)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE+(3,),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # freeze base

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# Train head
history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS)

# Optionally fine-tune some base layers
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=3)

# Save model
model.save(MODEL_SAVE_PATH, include_optimizer=False)

# Convert to TFLite (float16 quantization)
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_SAVE_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# For size/latency friendly quantization: float16 (good for Raspberry Pi with TFLite)
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open(TFLITE_FILENAME, "wb") as f:
    f.write(tflite_model)
print("TFLite model saved to:", TFLITE_FILENAME)

# Evaluate on validation set (Keras) and print metrics
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation accuracy: {val_acc:.4f}, loss: {val_loss:.4f}")

# Optional: run inference with tflite interpreter on a few images
import PIL
from pathlib import Path

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILENAME)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_tflite_inference(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    interpreter.set_tensor(input_details[0]['index'], arr.astype(input_details[0]['dtype']))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return out  # raw softmax scores

# Try a sample
sample_images = list(Path(os.path.join(DATA_DIR,"val", class_names[0])).glob("*.jpg"))[:3]
for p in sample_images:
    preds = run_tflite_inference(str(p))
    pred_idx = np.argmax(preds)
    print(p.name, "=>", class_names[pred_idx], preds)
Notes on dataset

Use a small dataset (few hundred images per class) for quick experimentation.

Kaggle has recyclable/waste classification datasets (search “waste classification” or “recycling” datasets). Organize into train/val/test folders.

How to convert and quantize for Raspberry Pi (extra guidance)
Use the TFLite converter with Optimize.DEFAULT and target_spec.supported_types = [tf.float16] for smaller model and faster inference on Pi. Optionally do full integer quantization requiring a representative dataset function (better performance on devices with TFLite-optimized kernels).

For integer quantization (if you want the smallest AND fastest on certain accelerators), include representative_dataset callback when converting.

Example representative dataset function:

def representative_data_gen():
    for images, _ in train_ds.take(100):
        # images come as batches
        yield [tf.cast(images, tf.float32)]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
Deployment steps (Raspberry Pi)
Copy recycle_classifier.tflite to Raspberry Pi (scp or USB).

Install tflite runtime or full TF if Pi supports it:

Preferred: pip install tflite-runtime (smaller).

Or: pip install tensorflow if supported on your Pi model.

Use Python script to capture camera frames (OpenCV) and run TFLite inference per frame.

Optimize: resize input to 224x224; use batching=1; consider threaded capture + inference.

Minimal inference loop (on Pi):

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="recycle_classifier.tflite")
interpreter.allocate_tensors()
in_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    img = cv2.resize(frame, (224,224))
    inp = np.expand_dims(img.astype(np.float32), axis=0)
    # preprocess if needed
    interpreter.set_tensor(in_details[0]['index'], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(out_details[0]['index'])
    idx = np.argmax(output)
    label = class_names[idx]
    # Draw label on frame
    cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow('Edge Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()
Report items to include (Deliverable)
Dataset description (classes, counts).

Model architecture (MobileNetV2 feature extractor + Dense head).

Training hyperparameters and results (train/val accuracy & loss plots).

Confusion matrix on validation/test set.

TFLite conversion details (quantization method).

On-device inference latency measurements (e.g., average ms per inference on Raspberry Pi).

Deployment steps and code snippets (capture, preprocess, run TFLite interpreter).

Limitations and future improvements (edge caching, model pruning, model update strategy).

Task 1 — Example evaluation & metrics (how to compute)
After training in the notebook you can compute:

Accuracy: val_acc from model.evaluate.

Precision/Recall/F1 using sklearn:

from sklearn.metrics import classification_report, confusion_matrix
# Collect true and predicted labels for val set
y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true += list(np.argmax(labels.numpy(), axis=1))
    y_pred += list(np.argmax(preds, axis=1))
print(classification_report(y_true, y_pred, target_names=class_names))
print(confusion_matrix(y_true, y_pred))
Record these in the report.

Task 2 — AI-Driven IoT Concept (Smart Agriculture Simulation)
Objective
Design a simulated smart agriculture system that uses sensors (IoT) to collect environmental and soil data and an AI model to predict short-term crop yield and irrigation needs.

Sensors needed
Soil moisture sensor (capacitance or resistive) — per zone

Soil temperature sensor

Air temperature sensor

Relative humidity sensor

Light (PAR) sensor or lux sensor

pH sensor (optional, to track soil acidity)

EC (electrical conductivity) sensor — for salinity/nutrient proxy (optional)

Rain gauge (optional)

Camera module (for crop health imagery, optional)

Flow meter (for irrigation water usage)

CO₂ sensor (optional for greenhouse)

Edge vs Cloud roles
Edge (on microcontroller / gateway): raw sampling, preprocessing, anomaly detection, immediate irrigation control (low-latency), data buffering.

Cloud / CPU gateway: heavier model training, historical aggregation, long-term yield prediction, model retraining.

Proposed AI model(s)
Short-term irrigation decision model: small decision-tree or gradient-boosted regressor running at the gateway. Input: soil moisture, weather forecast (local), evapotranspiration estimate, crop growth stage. Output: irrigation on/off and minutes.

Crop yield predictor (time series): hybrid model combining:

Feature engineering: cumulative degree-days, rainfall totals, average soil moisture, NDVI from images.

Model: LSTM/GRU or Temporal Convolutional Network for time-series features; or classical ensemble (XGBoost/LightGBM) on engineered features. Lightweight version for edge: a small MLP or tree model pre-trained and compressed.

Image-based disease detection (optional): small CNN (MobileNetV2) for leaf disease detection — runs on gateway or Raspberry Pi with camera.

Why hybrid? Time-series LSTM can capture temporal dependencies; tree models are robust and interpretable. For edge deployment, distilled/lightweight models or decision rules are preferable.

Data flow diagram (text/ASCII sketch)
[SENSORS: soil moisture, temp, humidity, light, pH, flow, camera]
        |   (sample at intervals, e.g., every 5-15 mins)
        V
[Local MCU / Sensor Node]
    - preprocess (filter, timestamp)
    - local alert rules (e.g., moisture < threshold -> immediate irrigation)
    - transmit to Gateway (MQTT over Wi-Fi/LoRaWAN)
        |
        V
[Gateway / Raspberry Pi]
    - aggregate sensor streams
    - run Edge AI models:
         * real-time irrigation recommender (fast)
         * image disease detector (if camera)
    - forward summarized data & model updates to Cloud (HTTPS/MQTT)
        |
        V
[Cloud]
    - store historical data (time-series DB)
    - train / retrain heavier ML models (yield predictor, forecasting)
    - expose API for dashboards & farm manager
        |
        V
[USER DASHBOARD / MOBILE APP]
    - visualizations, alerts, manual override, scheduling
Example flow of a prediction cycle
Sensors send data to gateway.

Gateway runs irrigation recommender (Edge model) → triggers irrigation controller if needed.

Gateway streams nightly aggregated data to cloud.

Cloud retrains yield predictor weekly (or on new labeled harvest data) and pushes small updated model weights to gateway (via OTA).

User views dashboard with predicted yield and recommendations.

Simulation notes (how to simulate)
Create a synthetic dataset or use public crop/weather datasets (e.g., NASA POWER, local weather sources) for historical weather and augment with plant growth model rules.

Simulate sensors by sampling real weather/time-series and adding noise.

Train models on historical features + simulated sensor noise then evaluate.

Example model spec (yield predictor)
Input features (per day): average soil moisture, avg temp, cumulative growing degree days, rainfall_7d, total_light, NDVI_mean (if camera), irrigation_volume.

Model: XGBoost regressor (works well on tabular features) or LSTM if sequence length matters.

Output: predicted yield per hectare (continuous).

Ethics & Challenges (brief reflection)
Privacy: sensor and camera data may include personally identifying info (neighbouring fields) — apply strict access control and on-device processing for images.

Bias & fairness: models trained on one region’s data may not generalize; ensure diversity in training data.

Security: IoT devices must be secured (OTA updates, encrypted comms, device authentication) to avoid malicious control.

Environmental impact: Edge devices reduce bandwidth but produce e-waste — design for longevity and recyclability.

Transparency: provide farmers with interpretable recommendations and override controls; avoid opaque automated decisions that cause crop loss.

Deliverables & GitHub Structure (Suggested)
ai-future-directions/
├─ README.md
├─ part1_theory/
│  ├─ essay_q1_q2.md
├─ part2_edge_ai/
│  ├─ notebook_training.ipynb
│  ├─ train.py
│  ├─ requirements.txt
│  ├─ saved_model/        # (optional: small sample saved model or link)
│  ├─ recycle_classifier.tflite  # (add after conversion)
│  ├─ inference_pi.py
│  ├─ dataset_description.md
├─ part2_iot_concept/
│  ├─ sensors_list.md
│  ├─ architecture_diagram.txt
│  ├─ simulation_script/  # optional scripts for data simulation
└─ report.pdf  # consolidated report (optional)
README.md should include

How to run the notebook in Colab.

How to convert model to TFLite.

How to deploy to Raspberry Pi (install tflite runtime, test script).

Citation of datasets used (Kaggle links).

License (e.g., MIT).

