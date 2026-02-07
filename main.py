import paho.mqtt.client as mqtt
import json
import numpy as np
import joblib 
import math 
import pandas as pd
import schedule 
import time 
import requests 
from datetime import datetime, time as dt_time
import warnings

# Mengabaikan peringatan saat loading model (misalnya UserWarning dari Scikit-learn)
warnings.filterwarnings("ignore", category=UserWarning)

# ====================================================================
# === KONFIGURASI PROYEK & KREDENSIAL ===
# ====================================================================

# --- MQTT Broker ---
# MQTT_BROKER_HOST = "74489fb16d844b74b1e9db42ef05d167.s1.eu.hivemq.cloud" 
# MQTT_BROKER_PORT = 8883 # Port SSL/TLS
# MQTT_USER = "firdian" 
# MQTT_PASS = "F1rdianr" 
MQTT_BROKER_HOST = "192.168.8.100" 
MQTT_BROKER_PORT = 1883 # Port SSL/TLS
MQTT_USER = "bangfir" 
MQTT_PASS = "B4ngfir!" 

# --- KONFIGURASI PUSHOVER ---
PUSHOVER_USER_KEY = "u9b3snj1g6w8ix3mnjbwghb5x7bdsu" 
PUSHOVER_API_TOKEN = "air9kmrv3hp8w47kwi2ou3omnecbrb"

# --- Topik MQTT ---
MQTT_TOPIC_SUBSCRIBE = "kanopi/sensor" 
MQTT_TOPIC_PUBLISH = "kanopi/status/prediksi" 
TOPIC_PERINTAH_MOTOR = "kanopi/perintah/motor" # Untuk mengirim perintah tutup saat malam
TOPIC_STATUS_POSISI = "kanopi/status/posisi"
TOPIC_STATUS_AUTO = "kanopi/status/auto"

# --- PARAMETER SMOOTHING ---
PREDICTION_BUFFER_SIZE = 7 
PREDICTION_HISTORY = []    
LAST_REPORTED_STATUS = "cerah" 
LAST_DAILY_TRIGGER_DATE = None 

# --- URUTAN FITUR ---
FEATURES_ORDER = ['suhu', 'kelembapan', 'cahaya', 'hour_sin', 'hour_cos']

# --- PARAMETER WAKTU HARIAN ---
MORNING_TRIGGER_HOUR = 6
MORNING_TRIGGER_MINUTE = 0
EVENING_TRIGGER_HOUR = 17
EVENING_TRIGGER_MINUTE = 30

# ====================================================================
# === MUAT MODEL & PARAMETER NORMALISASI ===
# ====================================================================

try:
    # Muat Model Random Forest (.pkl)
    model = joblib.load('random_forest_model.pkl')
    
    # Muat Parameter Normalisasi (.json)
    with open('normalization_params.json', 'r') as f:
        NORMALIZATION_PARAMS = json.load(f)
        
    print("✅ Model RF dan Parameter Normalisasi berhasil dimuat.")
except FileNotFoundError as e:
    print(f"❌ ERROR: File model/scaler tidak ditemukan. Pastikan file ada di folder yang sama: {e}")
    exit()

# ====================================================================
# === FUNGSI HELPER & LOGIKA INTI ===
# ====================================================================

def get_cyclic_features(current_datetime):
    """Menghitung hour_sin dan hour_cos berdasarkan waktu server."""
    current_hour= current_datetime.hour
    
    jam_sinus = math.sin(current_hour* 15 - 90)
    jam_cosinus = math.cos(current_hour* 15 - 90)
    
    return jam_sinus, jam_cosinus

def send_pushover_notification(title, message):
    """Mengirim notifikasi ke Pushover menggunakan HTTP POST."""
    url = "https://api.pushover.net/1/messages.json"
    data = {
        "token": PUSHOVER_API_TOKEN, 
        "user": PUSHOVER_USER_KEY,
        "title": title,
        "message": message
    }
    try:
        requests.post(url, data=data)
        print(f"   [NOTIFICATION] Notifikasi Pushover terkirim: {title}")
    except Exception as e:
        print(f"   [NOTIFICATION] Gagal mengirim Pushover: {e}")

def scale_value(value, feature_name):
    """Normalisasi nilai input tunggal (MinMaxScaler replication)."""
    params = NORMALIZATION_PARAMS[feature_name]
    min_val = params['min']
    max_val = params['max']
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def preprocess_and_predict(payload_json):
    """Memproses data, menambahkan fitur waktu SERVER, dan menjalankan inferensi."""
    
    data = json.loads(payload_json)
    current_time = datetime.now() 
    jam_sinus, jam_cosinus = get_cyclic_features(current_time)
    
    processed_features = []
    sensor_names = ['suhu', 'kelembapan', 'cahaya']
    
    # Normalisasi 3 Fitur Sensor
    for name in sensor_names:
        processed_features.append(scale_value(data[name], name))
        
    # Tambahkan 2 Fitur Waktu (dihitung dari Server)
    processed_features.append(jam_sinus)
    processed_features.append(jam_cosinus)

    # Ubah ke array numpy (shape 1, 5)
    processed_input = np.array(processed_features).reshape(1, -1)
    
    # Inferensi
    prediction_label = model.predict(processed_input)[0]
    result_text = "mendung" if prediction_label == 1 else "cerah"
    
    return result_text, int(prediction_label)

def run_prediction_and_smooth(payload_json):
    """smoothing, dan memicu notifikasi jika terjadi perubahan tren."""
    global PREDICTION_HISTORY, LAST_REPORTED_STATUS

    current_time = datetime.now()
    
    # Kunci: HANYA jalankan ML jika waktu berada dalam jam operasional
    # Note: Kita tetap jalankan prediksi di malam hari untuk mengisi buffer, 
    # tetapi tidak mengirim update status yang di-smoothing/kontroversial.
    
    # 1. Jalankan prediksi real-time
    result_text, prediction_label = preprocess_and_predict(payload_json)

    # 2. Tambahkan hasil prediksi biner ke history
    PREDICTION_HISTORY.append(prediction_label)
    if len(PREDICTION_HISTORY) > PREDICTION_BUFFER_SIZE:
        PREDICTION_HISTORY.pop(0) 
    
    print(f"Buffer Hasil Prediksi: {PREDICTION_HISTORY}")

    # Logika Voting hanya relevan selama jam operasional
    if current_time.hour >= MORNING_TRIGGER_HOUR and (current_time.hour < EVENING_TRIGGER_HOUR or (current_time.hour == EVENING_TRIGGER_HOUR and current_time.minute < EVENING_TRIGGER_MINUTE)):
        
        if len(PREDICTION_HISTORY) == PREDICTION_BUFFER_SIZE:
            mendung_count = sum(PREDICTION_HISTORY) 
            cerah_count = PREDICTION_BUFFER_SIZE - mendung_count
            
            current_smoothed_status = "mendung" if mendung_count > cerah_count else "cerah"
            
            # Kirim status prediksi yang sudah di-smoothing ke MQTT
            client.publish(MQTT_TOPIC_PUBLISH, json.dumps({
                "status": current_smoothed_status, 
                "label": prediction_label
            }), qos=1, retain=True)

            # 3. Cek Logika Notifikasi Perubahan Tren (di tengah hari)
            if current_smoothed_status != LAST_REPORTED_STATUS:
                # Tren: CERAH -> MENDUNG
                if current_smoothed_status == "mendung" and LAST_REPORTED_STATUS == "cerah":
                    if CURRENT_AUTO_MODE_STATUS == "true":
                        title = "🚨 Peringatan Cuaca Mendung"
                        message = f"Model telah mengonfirmasi cuaca mendung. Kanopi akan ditutup."
                    else:
                        title = "🚨 Peringatan Cuaca Mendung"
                        message = f"Model telah mengonfirmasi cuaca mendung. Disarankan untuk menutup kanopi atau mengaktifkan mode otomatis."
                    #send_pushover_notification(title, message)
                    
                # Tren: MENDUNG -> CERAH
                elif current_smoothed_status == "cerah" and LAST_REPORTED_STATUS == "mendung":
                    if CURRENT_AUTO_MODE_STATUS == "true":
                        title = "☀️ Cuaca Cerah"
                        message = f"Model mengonfirmasi cuaca cerah. Kanopi akan dibuka."
                    else:
                        title = "☀️ Cuaca Cerah"
                        message = f"Model mengonfirmasi cuaca cerah. Anda dapat membuka kanopi."
                    #send_pushover_notification(title, message)
                
                # Update status terakhir yang dilaporkan
                LAST_REPORTED_STATUS = current_smoothed_status
            
            return current_smoothed_status
    
    return "Not Operating Hour"

# ====================================================================
# === FUNGSI SCHEDULER HARIAN ===
# ====================================================================

def daily_morning_trigger():
    """Dipanggil tepat pukul 06:00. Menganalisis buffer untuk status CERAH pagi."""
    global PREDICTION_HISTORY, LAST_REPORTED_STATUS, LAST_DAILY_TRIGGER_DATE
    
    current_date = datetime.now().date()
    # Pastikan trigger hanya berjalan sekali per hari
    if current_date == LAST_DAILY_TRIGGER_DATE:
        return
    
    # Ambil voting dari buffer terakhir (bisa saja dari data malam, tetapi ini adalah best effort)
    if len(PREDICTION_HISTORY) < PREDICTION_BUFFER_SIZE:
        current_smoothed_status = "cerah" # Default aman di pagi hari
    else:
        mendung_count = sum(PREDICTION_HISTORY)
        cerah_count = PREDICTION_BUFFER_SIZE - mendung_count
        current_smoothed_status = "mendung" if mendung_count > cerah_count else "cerah"

    # Notifikasi Pagi (Memicu notifikasi dan status awal)
    if current_smoothed_status == "cerah":
        title = "☀️ Laporan Pagi: Cerah (Mode Aktif)"
        message = "Model mengonfirmasi cuaca cerah di pagi hari. Kanopi dapat dibuka atau setel mode otomatis."
        send_pushover_notification(title, message)
        LAST_REPORTED_STATUS = "cerah"
    else:
        title = "☁️ Laporan Pagi: Mendung/Berawan"
        message = "Model mendeteksi awan di pagi hari. Kanopi disarankan tetap tertutup."
        send_pushover_notification(title, message)
        LAST_REPORTED_STATUS = "mendung"
        
    # Kirim status awal ke MQTT
    client.publish(MQTT_TOPIC_PUBLISH, json.dumps({"status": current_smoothed_status, "label": (1 if current_smoothed_status == 'mendung' else 0)}), qos=1, retain=True)
    
    LAST_DAILY_TRIGGER_DATE = current_date
    PREDICTION_HISTORY = [] # Kosongkan buffer untuk pengumpulan data siang


def daily_evening_trigger():
    """Dipanggil tepat pukul 17:30. Mengirim notifikasi tutup paksa jika kanopi terbuka."""
    global LAST_REPORTED_STATUS, PREDICTION_HISTORY, CURRENT_CANOPY_STATUS
    
    # Logika untuk mendapatkan status kanopi terakhir
    # NOTE: Kita harus mendapatkan status ini dari retained message MQTT
    # Cek status terakhir dari broker (asumsi variabel global CURRENT_CANOPY_STATUS ter-update via on_status_message)
    
    if LAST_REPORTED_STATUS == "cerah" or CURRENT_CANOPY_STATUS == "terbuka":
        
        title = "🌙 Peringatan Tutup: Sudah Malam"
        message = "Kanopi akan ditutup karena sudah melewati jam 17:30. Mode pengawasan ML dinonaktifkan."
        send_pushover_notification(title, message)
        
        # Kirim perintah penutupan otomatis (opsional, jika Anda ingin kanopi otomatis menutup)
        client.publish(TOPIC_PERINTAH_MOTOR, "tutup", qos=1) 
        
    else:
        print(f"   [SCHEDULER] Jam 17:30: Kanopi sudah tertutup. Notifikasi diabaikan.")
    
    # Set status aman untuk malam hari dan kirim ke MQTT
    LAST_REPORTED_STATUS = "mendung" 
    PREDICTION_HISTORY = [] 
    client.publish(MQTT_TOPIC_PUBLISH, json.dumps({"status": "mendung", "label": 1}), qos=1, retain=True)


# ====================================================================
# === MQTT CALLBACKS & MAIN PROGRAM ===
# ====================================================================

# Global client object
client = mqtt.Client(transport="tcp")

def on_status_message(client, userdata, msg):
    """Callback khusus untuk menyimpan status posisi kanopi (terbuka/tertutup)."""
    global CURRENT_CANOPY_STATUS
    try:
        payload = msg.payload.decode()
        # Payload harus berupa string sederhana: "terbuka" atau "tertutup"
        CURRENT_CANOPY_STATUS = payload.lower().strip()
    except Exception as e:
        print(f"   ❌ Error updating canopy status: {e}")
        
def on_status_auto_message(client, userdata, msg):
    """Callback khusus untuk menyimpan status mode otomatis (true/false)."""
    global CURRENT_AUTO_MODE_STATUS
    try:
        payload = msg.payload.decode()
        CURRENT_AUTO_MODE_STATUS = payload.lower().strip()
    except Exception as e:
        print(f"   ❌ Error updating auto mode status: {e}")

def on_connect(client, userdata, flags, rc):
    """Dipanggil ketika klien terhubung ke broker."""
    if rc == 0:
        print(f"✅ Terhubung ke Broker MQTT ({MQTT_BROKER_HOST})")
        
        # Subscribe ke topic data mentah
        client.subscribe(MQTT_TOPIC_SUBSCRIBE)
        
        # Subscribe ke topic status posisi (Untuk mendapatkan status 'terbuka'/'tertutup')
        client.subscribe(TOPIC_STATUS_POSISI) 
        client.message_callback_add(TOPIC_STATUS_POSISI, on_status_message)
        
        client.subscribe(TOPIC_STATUS_AUTO) 
        client.message_callback_add(TOPIC_STATUS_AUTO, on_status_auto_message)

    else:
        print(f"❌ Gagal terhubung, kode: {rc}")

def on_message(client, userdata, msg):
    """Dipanggil ketika pesan data mentah atau pesan non-status lainnya diterima."""
    # Pastikan pesan bukan berasal dari topic status posisi (karena sudah di-handle oleh on_status_message)
    if msg.topic == TOPIC_STATUS_POSISI:
        return
    
    if msg.topic == TOPIC_STATUS_AUTO:
        return

    try:
        payload = msg.payload.decode()
        current_time = datetime.now()
        
        # Log penerimaan hanya selama jam operasional untuk mengurangi spam log
        if current_time.hour >= MORNING_TRIGGER_HOUR and current_time.hour <= EVENING_TRIGGER_HOUR:
            print(f"\n[{current_time.strftime('%H:%M:%S')}] RECEIVED: {payload}")
            
        # Jalankan prediksi dan smoothing
        smoothed_status = run_prediction_and_smooth(payload)
        
        if smoothed_status != "Not Operating Hour":
            print(f"   [SMOOTHED STATUS]: {smoothed_status.upper()} (Buffer: {len(PREDICTION_HISTORY)}/{PREDICTION_BUFFER_SIZE})")
        
    except Exception as e:
        print(f"❌ Error saat memproses pesan: {e}")


if __name__ == "__main__":
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    # comment jika akan menggunakan broker lokal tanpa TLS
    # client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
    
    client.on_connect = on_connect
    client.on_message = on_message

    # Jadwal Trigger Pagi (06:00)
    schedule_time_morning = f"{MORNING_TRIGGER_HOUR:02}:{MORNING_TRIGGER_MINUTE:02}"
    schedule.every().day.at(schedule_time_morning).do(daily_morning_trigger)
    
    # Jadwal Trigger Sore (17:30)
    schedule_time_evening = f"{EVENING_TRIGGER_HOUR:02}:{EVENING_TRIGGER_MINUTE:02}"
    schedule.every().day.at(schedule_time_evening).do(daily_evening_trigger)

    print(f"Scheduler aktif: Pagi di {schedule_time_morning}, Sore di {schedule_time_evening}")
    print(f"Mencoba terhubung ke MQTT Broker...")
    
    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        
        while True:
            client.loop() 
            schedule.run_pending() 
            time.sleep(1) 
            
    except Exception as e:
        print(f"❌ Gagal koneksi atau error loop: {e}")