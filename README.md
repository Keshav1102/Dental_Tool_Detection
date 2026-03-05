# Dental Tool Detection – Real‑time YOLO Inference Pipeline

This project provides a complete real‑time object detection pipeline for dental surgical tools. It consists of a **Next.js frontend**, an **Express.js backend**, and a **FastAPI‑based ML service** running a trained YOLO model. All communication is handled via **Socket.IO** for low‑latency, bidirectional messaging.

The frontend captures live camera frames and sends them to the backend. The backend forwards each frame to the ML service, which runs inference, draws bounding boxes directly on the image, and returns the annotated frame. The frontend then displays the annotated video stream, ensuring that bounding boxes are perfectly aligned with the frame they were computed on.

---

## System Architecture
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Next.js │─────▶│ Express │─────▶│ FastAPI │
│ Frontend │◀─────│ Backend │◀─────│ ML Service │
└─────────────┘ └─────────────┘ └─────────────┘ 
(port 3000) (port 5000) (port 8000)

text

- **Frontend (Next.js)** – captures video frames, converts them to base64, and emits them via Socket.IO. It receives annotated images and displays them.
- **Express Backend** – acts as a proxy. It assigns a unique `requestId` to each frame, forwards it to the ML service, and routes responses back to the correct frontend client.
- **ML Service (FastAPI + Socket.IO)** – loads a YOLO model, runs inference on incoming frames, draws bounding boxes and labels, and emits the annotated image (base64) along with detection metadata.

All components communicate over **WebSocket** (via Socket.IO) for minimal latency.

---

## Prerequisites

- **Node.js** (v18 or later)
- **Python** (3.10 or later)
- **npm** or **yarn**
- A trained **YOLO model** (`.pt` file) for dental tools (e.g., `best.pt` or `last.pt`)
- (Optional) NVIDIA GPU with CUDA for accelerated inference

---

## Installation

### 1. Clone or create the project structure
dental-tool-detection/
├── frontend/ # Next.js app
├── backend/ # Express server
└── ml-api/ # FastAPI + YOLO service

text

### 2. Set up the ML API (FastAPI)

Navigate to the `ml-api` directory.

- Install Python dependencies:
pip install fastapi uvicorn python-socketio opencv-python-headless numpy ultralytics

text
- Create a folder `model` and place your trained `.pt` file inside (e.g., `model/best.pt`).
- Create the inference module (`inference.py`) that defines a `YOLOModel` class for loading the model and running predictions.
- Create the main FastAPI app (`main.py`) that:
- Initialises a Socket.IO server.
- Loads the model on startup.
- Defines Socket.IO event handlers for `connect`, `disconnect`, and `frame`.
- In the `frame` handler, the image is decoded, inference is run, bounding boxes are drawn, and the annotated image is encoded back to base64 and emitted to the client (the Express backend) along with the original `requestId`.

### 3. Set up the Express backend

Navigate to the `backend` directory.

- Initialise a Node.js project and install dependencies:
npm init -y
npm install express cors socket.io socket.io-client

text
- Create `server.js` with the following logic:
- Create an Express app and HTTP server.
- Attach a Socket.IO server for frontend connections (CORS enabled).
- Create a Socket.IO client to connect to the ML API (`http://localhost:8000`).
- Maintain a `pendingRequests` map to associate `requestId` with frontend socket IDs.
- On `frame` from frontend, generate a `requestId`, store it, and forward the frame to the ML API.
- On `detections` from the ML API, forward the entire data object (which includes `annotated_image` and `detections`) to the corresponding frontend client.
- Handle disconnections and clean up pending requests.

### 4. Set up the Next.js frontend

Navigate to the `frontend` directory.

- Create a new Next.js app (if not already present):
npx create-next-app@latest . --use-npm

text
- Install Socket.IO client:
npm install socket.io-client

text
- Create a page (e.g., `app/page.js`) that:
- Uses `useRef` to access a hidden video element and canvas for frame capture.
- Uses `useState` for connection status, FPS, and the annotated image.
- Connects to the Express backend via Socket.IO.
- Listens for `detections` events and updates the `annotatedImage` state.
- Captures frames at a set FPS and emits them as `frame` events.
- Renders the annotated image using an `<img>` tag when available, otherwise shows a waiting message.

---

## Running the System

Start each component in a separate terminal, in the following order:

1. **ML API**
 ```bash
 cd ml-api
 uvicorn main:app --host 0.0.0.0 --port 8000
You should see ✅ ML API: model loaded. and Socket.IO connection logs.

Express Backend

bash
cd backend
node server.js
Expected output: ✅ Connected to ML API and 🚀 Express backend running on http://localhost:5000.

Next.js Frontend

bash
cd frontend
npm run dev
Open http://localhost:3000 in your browser.

Once the frontend loads, grant camera permission. You will see the live video feed replaced by an annotated stream once detections start arriving.

Usage
Hold a dental tool (scalpel, drill, etc.) in front of the camera.

If the model detects it with sufficient confidence, a green bounding box with the tool name and confidence will appear on the displayed image.

Use the slider to adjust the frames per second (FPS) sent to the backend. Lower FPS reduces network load but increases latency.

The connection status is shown at the top.

Troubleshooting
Symptom	Possible Cause	Solution
Frontend shows "Waiting for annotated frames..."	No detections received	Check browser console for WebSocket errors; verify Express and ML API are running.
ML API model fails to load	PyTorch 2.6 weights_only security	Add torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel]) before loading.
No bounding boxes appear	Confidence threshold too high; wrong model path; object not in training set	Lower threshold temporarily; verify model path; test with a known image.
Annotated image not displayed in frontend	Express drops annotated_image field	Ensure Express forwards the entire data object from ML API, not just detections.
High latency / lag	Network round trips; inference time on CPU	Reduce FPS; use GPU for inference; optimise model (ONNX/TensorRT).
Performance Optimisation
GPU acceleration: Ensure PyTorch with CUDA is installed and the model is moved to GPU (model.to('cuda')).

Model export: Convert .pt to ONNX or TensorRT for faster inference.

Batch processing: If multiple frames arrive quickly, the ML API could process them in a batch (requires changes).

Lower resolution: Resize frames to a smaller size (e.g., 320x320) before sending to reduce bandwidth and inference time.

Future Improvements
Add authentication and rate limiting.

Support multiple camera streams.

Implement model selection (e.g., different tools) via frontend.

Add recording and playback of detections.

Deploy using Docker and orchestration tools (Kubernetes, docker-compose).

License
This project is for educational and research purposes. Ensure you comply with any licenses for the YOLO model and training data used.
