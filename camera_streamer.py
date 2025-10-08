import cv2
import socketio
import time
import sys
import threading
from picamera2 import Picamera2
from libcamera import controls

class PiCameraStreamer:
    """
    Sebuah class untuk mengelola streaming video dari PiCamera ke server Socket.IO
    dalam sebuah thread terpisah.
    """
    def __init__(self, server_url, device_id, frame_rate=20):
        self.server_url = server_url
        self.device_id = device_id
        self.frame_rate = frame_rate

        self.picam2 = None
        self.sio = socketio.Client()
        self.thread = None
        self._is_running = False

        self._setup_sio_events()

    def _setup_camera(self):
        """Menginisialisasi dan mengkonfigurasi PiCamera."""
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(main={"size": (640, 480)})
            self.picam2.configure(config)
            self.picam2.start()
            print("✅ Kamera berhasil diinisialisasi.")
            
            # Coba atur autofocus
            try:
                self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                print("✅ Autofocus mode 'Continuous' diaktifkan.")
            except Exception:
                print("ℹ️ Autofocus tidak tersedia atau gagal diatur.")
            
            time.sleep(2) # Waktu pemanasan kamera
        except Exception as e:
            print(f"❌ Gagal menginisialisasi kamera: {e}")
            sys.exit(1)

    def _setup_sio_events(self):
        """Mendefinisikan event handlers untuk Socket.IO."""
        @self.sio.event
        def connect():
            print(f"✅ Berhasil terhubung ke server dengan ID: {self.device_id}")

        @self.sio.event
        def connect_error(data):
            print(f"❌ Gagal terhubung ke server: {data}")

        @self.sio.event
        def disconnect():
            print("🔌 Terputus dari server.")

    def _send_frames_loop(self):
        """Loop utama yang berjalan di thread untuk mengirim frame."""
        while self._is_running and self.sio.connected:
            try:
                frame = self.picam2.capture_array()
                is_success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if is_success:
                    self.sio.emit('frame_from_pi', buffer.tobytes())
                self.sio.sleep(1 / self.frame_rate)
            except Exception as e:
                print(f"Error saat mengirim frame: {e}")
                self._is_running = False # Hentikan loop jika ada error

    def start(self):
        """Memulai koneksi dan thread streaming."""
        if self._is_running:
            print("Streamer sudah berjalan.")
            return

        print("🚀 Memulai streamer...")
        self._setup_camera()
        
        try:
            connect_url = f"{self.server_url}?device_id={self.device_id}"
            self.sio.connect(connect_url, transports=['websocket'])
            
            self._is_running = True
            self.thread = threading.Thread(target=self._send_frames_loop)
            self.thread.start()
            print("🚀 Thread streaming dimulai.")
        except socketio.exceptions.ConnectionError as e:
            print(f"❌ Error koneksi fatal: {e}")
            self.picam2.stop() # Pastikan kamera berhenti jika koneksi gagal

    def stop(self):
        """Menghentikan thread streaming dan membersihkan sumber daya."""
        if not self._is_running:
            return

        print("🔌 Menghentikan streamer...")
        self._is_running = False
        if self.thread:
            self.thread.join() # Tunggu thread selesai
        
        if self.sio.connected:
            self.sio.disconnect()
        
        if self.picam2:
            self.picam2.stop()
        
        print(" Kamera dan koneksi ditutup.")