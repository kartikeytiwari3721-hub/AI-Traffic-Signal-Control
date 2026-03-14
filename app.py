from flask import Flask, render_template, Response, jsonify
from web_yolo import generate_frames
from web_sim import start_simulation_thread, sim_data
import os

app = Flask(__name__)

# Start the background traffic simulation thread
start_simulation_thread()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/simulation_stats')
def simulation_stats():
    return jsonify(sim_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
