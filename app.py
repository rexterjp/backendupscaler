from flask import Flask, request, jsonify, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_talisman import Talisman
import replicate
import os
import uuid
import requests
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Limiter
limiter = Limiter(
    get_remote_address,
    app=None,  # Initialize later if app is not created yet
    default_limits=["200 per day", "50 per hour"]
)

# Periksa apakah API token tersedia
replicate_api_token = os.getenv('REPLICATE_API_TOKEN')
if not replicate_api_token:
    print("PERINGATAN: REPLICATE_API_TOKEN tidak ditemukan di .env file!")
else:
    print(f"Menggunakan Replicate API token: {replicate_api_token[:5]}...{replicate_api_token[-5:]}")

app = Flask(__name__)
limiter.init_app(app) # Initialize Limiter with the app instance
CORS(app)

# Initialize Talisman with default secure settings
Talisman(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@limiter.limit("5 per minute") # Apply rate limit to this specific route
@app.route('/api/upscale', methods=['POST'])
def upscale_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Ambil parameter
    scale = request.form.get('scale', '4')
    face_enhance = request.form.get('face_enhance', 'true').lower() == 'true'

    # Proses file
    file = request.files['image']
    filename = str(uuid.uuid4()) + '.png'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Buka file yang sudah diupload dan kirimkan ke Replicate API langsung
        with open(filepath, "rb") as img_file:
            # Konversi scale ke float lalu ke int
            scale_value = int(float(scale))
            scale_value = max(1, min(scale_value, 10))

            # Tambahkan handling eksepsi khusus untuk replicate.run
            try:
                # Coba dapatkan ID model terbaru
                model_id = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"

                print(f"Memanggil Replicate API dengan model: {model_id}")

                # Jalankan proses upscale
                prediction = replicate.run(
                    model_id,
                    input={
                        "image": img_file,
                        "scale": scale_value,
                        "face_enhance": face_enhance
                    }
                )

                # Debug log
                print(f"Tipe output: {type(prediction)}")
                print(f"Output mentah dari Replicate API: {prediction}")

                # Set output untuk diproses selanjutnya
                output = prediction

                # Jika menggunakan library replicate versi baru, mungkin ada metode get_url()
                if hasattr(prediction, 'get_url'):
                    try:
                        output_url = prediction.get_url()
                        print(f"URL dari metode get_url(): {output_url}")
                        # Langsung lanjut ke proses download
                        response = requests.get(output_url)
                        if response.status_code == 200:
                            result_path = os.path.join(RESULTS_FOLDER, filename)
                            with open(result_path, 'wb') as f:
                                f.write(response.content)

                            # Dapatkan dimensi gambar
                            img = Image.open(result_path)
                            width, height = img.size

                            return jsonify({
                                'success': True,
                                'result_url': f'/api/results/{filename}',
                                'width': width,
                                'height': height,
                                'original_filename': file.filename,
                                'scale': scale_value
                            })
                    except Exception as url_error:
                        print(f"Gagal mendapatkan URL dengan get_url(): {str(url_error)}")
                        # Lanjutkan dengan cara lain

            except Exception as replicate_error:
                print(f"Error saat memanggil Replicate API: {str(replicate_error)}")

                # Jika error menunjukkan bahwa proses sedang berjalan atau sudah selesai
                error_str = str(replicate_error).lower()
                if "already exists" in error_str or "currently running" in error_str:
                    return jsonify({'error': 'Proses upscale sedang berjalan atau sudah selesai. Silakan cek history di akun Replicate Anda.'}), 500

                return jsonify({'error': f'Kesalahan Replicate API: {str(replicate_error)}'}), 500

        # Debugging: Log output dari Replicate API
        print(f"Output dari Replicate API: {type(output)}. Isi: {output}")

        # Pastikan output selalu dalam format URL string
        if isinstance(output, list) and len(output) > 0:
            # Format lama: output adalah list, ambil item pertama
            output_url = output[0]
        elif isinstance(output, str):
            # Format lama: output adalah string URL
            output_url = output
        elif isinstance(output, dict):
            # Format baru: output adalah dictionary
            if 'url' in output:
                output_url = output['url']
            elif 'output' in output and isinstance(output['output'], str):
                output_url = output['output']
            elif 'output' in output and isinstance(output['output'], list) and len(output['output']) > 0:
                output_url = output['output'][0]
            else:
                # Mencoba mencari URL dalam dictionary secara rekursif
                found = False
                for key, value in output.items():
                    if isinstance(value, str) and (value.startswith('http://') or value.startswith('https://')):
                        output_url = value
                        found = True
                        break
                if not found:
                    print(f"Tidak dapat menemukan URL dalam output: {output}")
                    return jsonify({'error': 'Format output dari Replicate API tidak dikenali'}), 500
        else:
            # Log untuk debugging
            print(f"Format output tidak dikenali: {type(output)}. Isi: {output}")
            # Coba konversi ke string - ini kemungkinan akan berhasil dengan FileOutput
            try:
                if hasattr(output, '__str__'):
                    output_str = str(output)
                    if output_str.startswith('http'):
                        output_url = output_str
                        print(f"Berhasil mengkonversi output ke URL: {output_url}")
                    else:
                        return jsonify({'error': 'Format output dari Replicate API tidak dikenali'}), 500
                else:
                    return jsonify({'error': 'Format output dari Replicate API tidak dikenali (tidak bisa dikonversi)'}), 500
            except Exception as conversion_error:
                print(f"Gagal mengkonversi output: {str(conversion_error)}")
                return jsonify({'error': 'Format output dari Replicate API tidak dikenali (konversi gagal)'}), 500

        # Download hasil
        response = requests.get(output_url)
        if response.status_code == 200:
            result_path = os.path.join(RESULTS_FOLDER, filename)
            with open(result_path, 'wb') as f:
                f.write(response.content)

            # Dapatkan dimensi gambar
            img = Image.open(result_path)
            width, height = img.size

            return jsonify({
                'success': True,
                'result_url': f'/api/results/{filename}',
                'width': width,
                'height': height,
                'original_filename': file.filename,
                'scale': scale_value
            })
        else:
            return jsonify({'error': 'Failed to download result'}), 500
    except Exception as e:
        print(f"Error pada backend: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Bersihkan file upload setelah diproses
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as cleanup_error:
                print(f"Error saat membersihkan file: {str(cleanup_error)}")

@app.route('/api/results/<filename>', methods=['GET'])
def get_result(filename):
    return send_file(os.path.join(RESULTS_FOLDER, filename))

@app.route('/api/check-history', methods=['POST'])
def check_history():
    """Endpoint untuk memeriksa history di Replicate berdasarkan ID"""
    if not replicate_api_token:
        return jsonify({'error': 'API token tidak ditemukan'}), 500

    try:
        prediction_id = request.json.get('prediction_id')
        if not prediction_id:
            return jsonify({'error': 'Prediction ID tidak diberikan'}), 400

        # Get prediction by ID
        try:
            import replicate
            prediction = replicate.predictions.get(prediction_id)

            # Check prediction status
            status = prediction.status
            output = prediction.output

            if status == "succeeded":
                # Jika output berhasil, coba download
                if output:
                    # Tentukan output URL berdasarkan tipe output
                    if isinstance(output, list) and len(output) > 0:
                        output_url = output[0]
                    elif isinstance(output, str):
                        output_url = output
                    elif isinstance(output, dict) and 'url' in output:
                        output_url = output['url']
                    else:
                        # Jika tidak bisa menentukan URL
                        return jsonify({
                            'status': status,
                            'output': str(output),
                            'error': 'Tidak bisa menentukan URL output'
                        })

                    # Buat nama file
                    filename = f"history_{prediction_id}.png"

                    # Download hasil
                    response = requests.get(output_url)
                    if response.status_code == 200:
                        result_path = os.path.join(RESULTS_FOLDER, filename)
                        with open(result_path, 'wb') as f:
                            f.write(response.content)

                        # Return success dengan URL lokal
                        return jsonify({
                            'success': True,
                            'status': status,
                            'result_url': f'/api/results/{filename}'
                        })

                # Jika tidak bisa download, kembalikan informasi saja
                return jsonify({
                    'status': status,
                    'output': str(output)
                })
            else:
                # Jika belum selesai
                return jsonify({
                    'status': status,
                    'message': f'Prediction status: {status}'
                })

        except Exception as e:
            print(f"Error checking prediction: {str(e)}")
            return jsonify({'error': f'Kesalahan saat mengecek prediction: {str(e)}'}), 500

    except Exception as e:
        print(f"Error pada endpoint check-history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk memeriksa status backend dan versi model
@app.route('/api/status', methods=['GET'])
def check_status():
    try:
        # Periksa token API
        token_status = "OK" if replicate_api_token else "MISSING"

        # Info versi
        model_info = {
            "model": "nightmareai/real-esrgan",
            "version": "f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"
        }

        import sys

        return jsonify({
            'status': 'online',
            'api_token_status': token_status,
            'model_info': model_info,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'replicate_version': replicate.__version__ if hasattr(replicate, '__version__') else "unknown"
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import sys
    print(f"Starting Real-ESRGAN backend server on port 5000...")
    print(f"Python version: {sys.version}")
    print(f"Replicate version: {getattr(replicate, '__version__', 'unknown')}")
    app.run(host='0.0.0.0', port=5000, debug=True)
