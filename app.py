from flask import Flask, render_template, request, jsonify
import os
from scripts.drawing_logic import DrawingLogic

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Inisialisasi objek DrawingLogic
drawing_logic = DrawingLogic()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_canvas', methods=['POST'])
def process_canvas():
    try:
        # Terima data canvas dari frontend
        canvas_data = request.json.get('canvas_data')
        
        # Proses data canvas menggunakan DrawingLogic
        result = drawing_logic.process_canvas(canvas_data)
        
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
