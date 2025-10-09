# server.py

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, join_room
from flask_cors import CORS
import numpy as np
import cv2
from ultralytics import YOLO
import base64
import eventlet
import threading
import time

# --- Konfigurasi Kinerja ---
PROCESSING_FPS = 10

# --- Inisialisasi ---
eventlet.monkey_patch()
try:
    model = YOLO('pkkm.pt')
except Exception as e:
    print(f"❌ Gagal memuat model 'pkkm.pt'. Pastikan file model ada di folder yang sama.")
    print(f"Error: {e}")
    exit()

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'proyek-streaming-kamera-anda'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Variabel Global untuk Thread-safe Frame Sharing ---
# Struktur baru: { 'device_id_1': {'raspicam': frame_data, 'thermal': frame_data}, ... }
latest_frames = {}
frame_lock = threading.Lock()
connected_pis = {}


# --- Halaman Web & API ---
@app.route('/')
def dashboard():
    """Menampilkan halaman dashboard utama."""
    return render_template('dashboard.html')

@app.route('/<device_id>')
def viewer(device_id):
    """Menampilkan halaman viewer untuk perangkat tertentu."""
    return render_template('viewer.html', device_id=device_id)

@app.route('/api/devices')
def get_active_devices():
    """API untuk mendapatkan daftar perangkat yang aktif."""
    return jsonify(list(connected_pis.keys()))


# --- WebSocket Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Menangani koneksi baru dari Pi atau browser."""
    device_id = request.args.get('device_id')
    if device_id:
        with frame_lock:
            connected_pis[device_id] = request.sid
            # Siapkan buffer frame untuk perangkat baru dengan sub-dictionary
            latest_frames[device_id] = {}
        print(f"✅ Perangkat terhubung: {device_id}")
        socketio.emit('update_devices', list(connected_pis.keys()))
    else:
        print("✅ Browser terhubung")

@socketio.on('join_room')
def handle_join_room(device_id):
    """Memasukkan browser ke dalam 'room' spesifik perangkat."""
    print(f"🖥️  Browser bergabung ke room: {device_id}")
    join_room(device_id)

@socketio.on('frame_from_pi')
def handle_frame(data):
    """Menerima frame dari Pi dan menyimpannya berdasarkan tipenya."""
    device_id = None
    for id, sid in connected_pis.items():
        if sid == request.sid:
            device_id = id
            break
    if not device_id: return

    # Ekstrak tipe stream dan data frame dari payload
    stream_type = data.get('type') # 'raspicam' atau 'thermal'
    frame_bytes = data.get('data')

    if stream_type and frame_bytes:
        with frame_lock:
            # Pastikan sub-dictionary ada
            if device_id not in latest_frames:
                latest_frames[device_id] = {}
            # Simpan frame ke dalam tipe yang sesuai
            latest_frames[device_id][stream_type] = frame_bytes

@socketio.on('disconnect')
def handle_disconnect():
    """Menangani saat Pi atau browser terputus."""
    disconnected_id = None
    for device_id, sid in list(connected_pis.items()):
        if sid == request.sid:
            disconnected_id = device_id
            break
            
    if disconnected_id:
        with frame_lock:
            del connected_pis[disconnected_id]
            if disconnected_id in latest_frames:
                del latest_frames[disconnected_id]
        print(f"❌ Perangkat terputus: {disconnected_id}")
        socketio.emit('update_devices', list(connected_pis.keys()))
    else:
        print("❌ Browser terputus")


# --- Background Thread untuk Pemrosesan AI ---
def processing_thread_func():
    """Loop yang berjalan di latar belakang untuk memproses frame dan mengirimnya."""
    print("🚀 Thread pemrosesan AI dimulai.")
    while True:
        devices_to_process = list(connected_pis.keys())

        for device_id in devices_to_process:
            raspicam_frame_data = None
            thermal_frame_data = None
            
            with frame_lock:
                # Ambil frame terbaru untuk setiap tipe
                device_frames = latest_frames.get(device_id, {})
                raspicam_frame_data = device_frames.get('raspicam')
                thermal_frame_data = device_frames.get('thermal')

            processed_payload = {}

            # 1. Proses frame Raspicam dengan YOLO
            if raspicam_frame_data:
                try:
                    npimg = np.frombuffer(raspicam_frame_data, np.uint8)
                    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        results = model.predict(frame, verbose=False, device='cpu') # Paksa CPU untuk stabilitas
                        annotated_frame = results[0].plot()
                        _, buffer = cv2.imencode('.jpg', annotated_frame)
                        b64_string = base64.b64encode(buffer).decode('utf-8')
                        processed_payload['raspicam'] = b64_string
                except Exception as e:
                    print(f"Error memproses frame raspicam untuk {device_id}: {e}")

            # 2. Proses frame Thermal (hanya encode ke Base64)
            if thermal_frame_data:
                try:
                    # Frame termal sudah dalam format JPG dari klien, cukup encode
                    b64_string = base64.b64encode(thermal_frame_data).decode('utf-8')
                    processed_payload['thermal'] = b64_string
                except Exception as e:
                    print(f"Error memproses frame thermal untuk {device_id}: {e}")

            # 3. Kirim payload gabungan ke browser jika ada data yang diproses
            if processed_payload:
                socketio.emit('update_streams', processed_payload, room=device_id)

        socketio.sleep(1 / PROCESSING_FPS)


if __name__ == '__main__':
    print("==============================================")
    print("       Server Streaming Kamera & AI         ")
    print("==============================================")
    socketio.start_background_task(target=processing_thread_func)
    print("Server berjalan di http://0.0.0.0:8000")
    socketio.run(app, host='0.0.0.0', port=8000)