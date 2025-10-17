import base64
import threading

import cv2
import eventlet
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, join_room
from ultralytics import YOLO

# --- Inisialisasi ---

PROCESSING_FPS = 10

eventlet.monkey_patch()

try:
    model = YOLO('./model/pkkm.pt')
    print("✅ Model YOLO 'pkkm.pt' berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model YOLO 'pkkm.pt'. Pastikan file model ada di direktori yang sama.")
    print(f"   Error: {e}")
    exit()

# Inisialisasi aplikasi Flask dan SocketIO
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'proyek-streaming-kamera-anda'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Variabel Global untuk Manajemen Frame & Koneksi ---

# Struktur untuk menyimpan frame terbaru dari setiap perangkat
# Format: { 'device_id_1': {'raspicam': frame_data, 'thermal': frame_data}, ... }
latest_frames = {}
frame_lock = threading.Lock()  # Lock untuk sinkronisasi thread

# Dictionary untuk melacak Raspberry Pi yang terhubung
# Format: { 'device_id': 'session_id' }
connected_pis = {}


# --- Rute Halaman Web & API ---

@app.route('/')
def dashboard():
    """Menampilkan halaman dashboard utama yang berisi daftar perangkat aktif."""
    return render_template('dashboard.html')

@app.route('/<device_id>')
def viewer(device_id):
    """Menampilkan halaman viewer untuk melihat stream dari perangkat tertentu."""
    return render_template('viewer.html', device_id=device_id)

@app.route('/api/devices')
def get_active_devices():
    """API endpoint untuk mendapatkan daftar perangkat (Raspberry Pi) yang sedang terhubung."""
    return jsonify(list(connected_pis.keys()))


# --- Penanganan Event SocketIO ---

@socketio.on('connect')
def handle_connect():
    """Menangani koneksi baru dari Raspberry Pi atau browser."""
    device_id = request.args.get('device_id')
    connection_type = request.args.get('type')

    # Jika koneksi berasal dari Raspberry Pi
    if device_id and connection_type == 'pi':
        with frame_lock:
            connected_pis[device_id] = request.sid
            latest_frames[device_id] = {}
        print(f"✅ Perangkat Pi terhubung: {device_id} (SID: {request.sid})")
        # Beri tahu semua klien (dashboard) bahwa ada perangkat baru
        socketio.emit('update_devices', list(connected_pis.keys()))
    
    # Jika koneksi berasal dari browser yang ingin melihat stream
    elif device_id:
        print(f"🖥️  Browser terhubung untuk melihat perangkat: {device_id}")
    
    # Koneksi lain (misalnya, browser ke halaman dashboard)
    else:
        print("✅ Browser terhubung ke dashboard.")

@socketio.on('join_room')
def handle_join_room(device_id):
    """Memasukkan browser ke dalam 'room' spesifik agar hanya menerima stream dari perangkat itu."""
    print(f"🚪 Browser bergabung ke room: {device_id}")
    join_room(device_id)

@socketio.on('frame_from_pi')
def handle_frame(data):
    """Menerima frame video (raspicam/thermal) dari Raspberry Pi."""
    device_id = None
    # Cari device_id berdasarkan session ID (SID) pengirim
    for id, sid in connected_pis.items():
        if sid == request.sid:
            device_id = id
            break
    
    if not device_id:
        return

    stream_type = data.get('type')  # 'raspicam' atau 'thermal'
    frame_bytes = data.get('data')

    if stream_type and frame_bytes:
        with frame_lock:
            # Pastikan dictionary untuk device_id ada
            if device_id not in latest_frames:
                latest_frames[device_id] = {}
            # Simpan frame terbaru
            latest_frames[device_id][stream_type] = frame_bytes

@socketio.on('disconnect')
def handle_disconnect():
    """Menangani ketika koneksi dari Pi atau browser terputus."""
    disconnected_id = None
    
    # Cek apakah yang terputus adalah Raspberry Pi yang terdaftar
    for device_id, sid in list(connected_pis.items()):
        if sid == request.sid:
            disconnected_id = device_id
            break

    # Jika yang terputus adalah Pi, bersihkan data terkait
    if disconnected_id:
        with frame_lock:
            del connected_pis[disconnected_id]
            if disconnected_id in latest_frames:
                del latest_frames[disconnected_id]
        print(f"❌ Perangkat Pi terputus: {disconnected_id}")
        # Kirim daftar perangkat terbaru ke semua klien
        socketio.emit('update_devices', list(connected_pis.keys()))
    else:
        # Jika bukan Pi (berarti browser), cukup catat di log
        print("❌ Browser terputus.")


# --- Background Thread untuk Pemrosesan AI (YOLO) ---

def processing_thread_func():
    """Loop yang berjalan di latar belakang untuk memproses frame dan mengirimkannya ke browser."""
    print("🚀 Thread pemrosesan AI (YOLO) dimulai.")
    while True:
        # Salin daftar perangkat agar tidak terpengaruh perubahan saat iterasi
        devices_to_process = list(connected_pis.keys())

        for device_id in devices_to_process:
            raspicam_frame_data = None
            thermal_frame_data = None
            
            with frame_lock:
                # Ambil frame terbaru untuk setiap tipe kamera
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
                        # Jalankan prediksi YOLO pada frame
                        results = model.predict(frame, verbose=False, device='cpu')
                        # Gambar hasil deteksi (bounding box, label) pada frame
                        annotated_frame = results[0].plot()
                        # Encode frame yang sudah diproses ke format JPG lalu ke Base64
                        _, buffer = cv2.imencode('.jpg', annotated_frame)
                        b64_string = base64.b64encode(buffer).decode('utf-8')
                        processed_payload['raspicam'] = b64_string
                except Exception as e:
                    print(f"⚠️ Error saat memproses frame raspicam untuk {device_id}: {e}")

            # 2. Proses frame Thermal (hanya encode ke Base64)
            if thermal_frame_data:
                try:
                    # Frame termal sudah dalam format JPG dari Pi, cukup encode
                    b64_string = base64.b64encode(thermal_frame_data).decode('utf-8')
                    processed_payload['thermal'] = b64_string
                except Exception as e:
                    print(f"⚠️ Error saat memproses frame thermal untuk {device_id}: {e}")

            # 3. Kirim payload gabungan ke browser jika ada data yang diproses
            if processed_payload:
                socketio.emit('update_streams', processed_payload, room=device_id)

        # Tunggu sejenak sesuai FPS yang ditentukan untuk mengurangi beban CPU
        socketio.sleep(1 / PROCESSING_FPS)


# --- Main Execution ---

if __name__ == '__main__':
    print("==============================================")
    print("    Server Streaming Kamera & Deteksi YOLO    ")
    print("==============================================")
    
    # Jalankan thread pemrosesan AI di latar belakang
    socketio.start_background_task(target=processing_thread_func)
    
    print("🌐 Server berjalan di http://0.0.0.0:8000")
    # Jalankan aplikasi web server
    socketio.run(app, host='0.0.0.0', port=8000)