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
# Atur berapa kali per detik model AI akan dijalankan.
# Nilai antara 2-7 adalah awal yang baik untuk CPU.
PROCESSING_FPS = 5

# --- Inisialisasi ---
eventlet.monkey_patch()
model = YOLO('pkkm.pt')
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'proyek-streaming-kamera'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Variabel Global untuk Thread-safe Frame Sharing ---
# Dictionary untuk menyimpan frame terakhir dari setiap perangkat
latest_frames = {}
# Lock untuk memastikan tidak ada konflik saat mengakses dictionary di atas
frame_lock = threading.Lock()
# Dictionary untuk melacak perangkat Pi yang terhubung
connected_pis = {}


# --- Halaman Web & API ---
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/<device_id>')
def viewer(device_id):
    return render_template('viewer.html', device_id=device_id)

@app.route('/api/devices')
def get_active_devices():
    return jsonify(list(connected_pis.keys()))


# --- WebSocket Event Handlers ---
@socketio.on('connect')
def handle_connect():
    device_id = request.args.get('device_id')
    if device_id:
        with frame_lock:
            connected_pis[device_id] = request.sid
            latest_frames[device_id] = None # Siapkan buffer frame untuk perangkat baru
        print(f"✅ Pi terhubung: {device_id}")
        socketio.emit('update_devices', list(connected_pis.keys()))
    else:
        print("✅ Browser terhubung")

@socketio.on('join_room')
def handle_join_room(device_id):
    print(f"Browser bergabung ke room: {device_id}")
    join_room(device_id)

@socketio.on('frame_from_pi')
def handle_frame(data):
    """Fungsi ini sekarang SANGAT RINGAN. Hanya menerima dan menyimpan frame."""
    device_id = None
    for id, sid in connected_pis.items():
        if sid == request.sid:
            device_id = id
            break
    if not device_id: return

    # Simpan frame terbaru ke dalam dictionary secara thread-safe
    with frame_lock:
        latest_frames[device_id] = data

@socketio.on('disconnect')
def handle_disconnect():
    disconnected_id = None
    for device_id, sid in connected_pis.items():
        if sid == request.sid:
            disconnected_id = device_id
            break
            
    if disconnected_id:
        with frame_lock:
            del connected_pis[disconnected_id]
            if disconnected_id in latest_frames:
                del latest_frames[disconnected_id]
        print(f"❌ Pi terputus: {disconnected_id}")
        socketio.emit('update_devices', list(connected_pis.keys()))
    else:
        print("❌ Browser terputus")


# --- Background Thread untuk Pemrosesan AI ---
def processing_thread_func():
    """Loop yang berjalan di latar belakang untuk memproses frame."""
    print("🚀 Thread pemrosesan AI dimulai.")
    while True:
        # Salin daftar perangkat agar tidak bentrok dengan koneksi baru/putus
        devices_to_process = list(connected_pis.keys())

        for device_id in devices_to_process:
            frame_data = None
            with frame_lock:
                frame_data = latest_frames.get(device_id)

            if frame_data:
                try:
                    npimg = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Proses dengan YOLOv8
                        results = model.predict(frame, verbose=False) # YOLO dapat menerima RGB langsung dari picamera2
                        annotated_frame_rgb = results[0].plot() # Ini diasumsikan mengembalikan RGB

                        # KONVERSI KEMBALI DARI RGB KE BGR SEBELUM ENCODING DENGAN CV2
                        # Cv2.imencode mengharapkan gambar dalam format BGR.
                        annotated_frame_bgr = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)

                        # Encode dan kirim ke browser di room yang spesifik
                        _, buffer = cv2.imencode('.jpg', annotated_frame_bgr)
                        b64_string = base64.b64encode(buffer).decode('utf-8')
                        socketio.emit('processed_frame_to_browser', b64_string, room=device_id)

                except Exception as e:
                    print(f"Error di thread pemrosesan untuk {device_id}: {e}")

        # Tunggu sejenak sesuai FPS yang diinginkan untuk mengurangi beban CPU
        socketio.sleep(1 / PROCESSING_FPS)


if __name__ == '__main__':
    # Mulai thread pemrosesan AI sebagai background task
    socketio.start_background_task(target=processing_thread_func)
    # Jalankan server
    socketio.run(app, host='0.0.0.0', port=8000)