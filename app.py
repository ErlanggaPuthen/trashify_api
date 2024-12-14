import io
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import logging
import pymysql
pymysql.install_as_MySQLdb()

app = Flask(__name__)

# Logging untuk memantau aktivitas server
logging.basicConfig(level=logging.INFO)

# Path model .tflite
MODEL_PATH = 'model_sampah.tflite'

# Muat model TensorFlow Lite
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()  # Alokasikan tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info(f"Model '{MODEL_PATH}' berhasil dimuat.")
except Exception as e:
    logging.error(f"Gagal memuat model '{MODEL_PATH}': {e}")
    raise

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Melakukan pra-pemrosesan gambar untuk klasifikasi."""
    logging.info("Memulai preprocessing gambar.")
    image = image.convert("RGB").resize((150, 150))  # Ubah ukuran sesuai model
    image = np.array(image, dtype=np.float32) / 255.0  # Normalisasi
    preprocessed_image = np.expand_dims(image, axis=0)  # Tambahkan batch dimensi
    logging.info("Preprocessing selesai.")
    return preprocessed_image

def predict_class(processed_image: np.ndarray) -> dict:
    """Melakukan prediksi kelas dan kepercayaan dari gambar yang telah diproses."""
    logging.info("Melakukan prediksi dengan model TensorFlow Lite.")
    interpreter.set_tensor(input_details[0]['index'], processed_image)  # Masukkan input
    interpreter.invoke()  # Jalankan inferensi
    predictions = interpreter.get_tensor(output_details[0]['index'])  # Ambil output

    class_index = int(np.argmax(predictions, axis=1)[0])
    confidence = int(np.round(np.max(predictions) * 100))  # Dibulatkan ke integer
    logging.info(f"Prediksi selesai: Kelas - {class_index}, Kepercayaan - {confidence}%")
    return {'class': class_index, 'confidence': confidence}

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi gambar."""
    if 'file' not in request.files:
        logging.warning("Permintaan tidak berisi file gambar.")
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400

    file = request.files['file']
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        prediction = predict_class(processed_image)

        logging.info(f"Prediksi: Kelas - {prediction['class']}, Kepercayaan - {prediction['confidence']}%")
        return jsonify({
            'class': prediction['class'],  # Tetap integer
            'confidence': f"{prediction['confidence']}%"  # Ditambahkan '%' dalam respons
        })

    except Exception as e:
        logging.error(f"Kesalahan saat memproses gambar: {e}")
        return jsonify({'error': 'Gagal memproses gambar'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
