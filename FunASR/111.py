import sounddevice as sd
import numpy as np
import queue
import threading
from funasr import AutoModel
import time
import sys
import json
from flask import Flask, Response, render_template_string, jsonify, request
from flask_cors import CORS
import os

for key in list(os.environ.keys()):
    if "proxy" in key.lower():
        os.environ.pop(key, None)

app = Flask(__name__)
CORS(app)

# 全局变量
asr_engine = None
recognition_results = []
result_lock = threading.Lock()


class RealTimeMicASR:
    def __init__(self, sample_rate=16000, chunk_duration=2.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.cache = {}
        self.current_text = ""
        self.final_text = ""
        self.full_text = []
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.accumulated_text = ""
        self.last_output_text = ""
        self.silence_frames = 0
        self.vad_triggered = False
        self.frame_count = 0
        self.ready_to_flush = False
        self.audio_buffer = np.array([], dtype=np.float32)  # 累积音频缓冲

        print("正在加载模型（无VAD，使用静音检测）...")

        # ✅ 方案二：不使用VAD，只加载主模型
        self.model = AutoModel(
            model="./models/paraformer-zh",
            device="cuda:0",
            disable_update=True,
        )
        print("模型加载完成！")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"状态: {status}", file=sys.stderr)

        if indata.shape[1] > 1:
            audio_chunk = np.mean(indata, axis=1).astype(np.float32)
        else:
            audio_chunk = indata[:, 0].copy().astype(np.float32)

        self.audio_queue.put(audio_chunk)

    def process_audio(self):
        """主处理循环：累积音频 -> 识别 -> 智能输出"""
        audio_buffer = np.array([], dtype=np.float32)
        last_process_time = time.time()
        min_process_interval = 0.3
        silence_threshold = 0.005  # 静音阈值
        silence_duration = 0  # 静音持续时间

        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, chunk])
                self.frame_count += 1

                # 检测静音
                chunk_energy = np.sqrt(np.mean(chunk ** 2))
                if chunk_energy < silence_threshold:
                    silence_duration += len(chunk) / self.sample_rate
                else:
                    silence_duration = 0

                current_time = time.time()
                buffer_duration = len(audio_buffer) / self.sample_rate

                # 处理条件：达到chunk大小 或 有足够音频且静音达到阈值
                should_process = (
                        buffer_duration >= self.chunk_duration or
                        (buffer_duration >= 1.0 and current_time - last_process_time >= min_process_interval)
                )

                # 如果静音超过0.5秒且缓冲区有内容，强制处理
                if silence_duration >= 0.5 and buffer_duration >= 0.5:
                    should_process = True

                if should_process and buffer_duration >= 0.5:
                    process_audio = audio_buffer.copy()
                    audio_buffer = np.array([], dtype=np.float32)
                    silence_duration = 0

                    try:
                        # 调用模型识别
                        result = self.model.generate(
                            input=process_audio,
                            cache=self.cache,
                            is_final=False,
                        )

                        if result and len(result) > 0:
                            text = result[0].get("text", "").strip()
                            # 检查是否结束（通过静音判断）
                            is_final = silence_duration >= 0.5

                            if text:
                                self._handle_recognition_result(text, is_final)

                    except Exception as e:
                        print(f"\n处理错误: {e}")

                    last_process_time = current_time

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n处理错误: {e}")

    def _handle_recognition_result(self, text, is_final):
        """处理识别结果"""
        # 去除可能的重复片段
        if text == self.last_output_text:
            return

        # 累积文本
        self.accumulated_text += text
        self.last_output_text = text

        # 更新实时显示
        self.current_text = self.accumulated_text

        # 判断是否为完整句子
        if is_final or self._is_complete_sentence(self.accumulated_text):
            self._flush_accumulated_text()

    def _is_complete_sentence(self, text):
        """智能判断是否为完整句子"""
        if not text or len(text) < 3:
            return False

        # 1. 包含结束标点
        if any(p in text for p in ['。', '！', '？', '…']):
            return True

        # 2. 长度足够（至少12个字符）
        if len(text) >= 12:
            return True

        # 3. 包含明显的语气词或完整结构
        if any(word in text for word in ['吧', '啦', '呢', '呀', '啊']) and len(text) >= 4:
            return True

        return False

    def _flush_accumulated_text(self):
        """将累积的文本归档"""
        if not self.accumulated_text:
            return

        # 去除重复内容
        text_to_flush = self.accumulated_text.strip()
        if not text_to_flush:
            return

        # 检查是否与最后一条历史记录重复
        with result_lock:
            if recognition_results and recognition_results[-1].get("text", "") == text_to_flush:
                self.accumulated_text = ""
                self.current_text = ""
                return

            # 归档
            recognition_results.append({
                "time": time.strftime("%H:%M:%S"),
                "text": text_to_flush
            })

        # 清空缓冲区
        self.accumulated_text = ""
        self.current_text = ""
        self.last_output_text = ""

    def start(self, device_index=None):
        """启动录音"""
        self.is_recording = True
        self.accumulated_text = ""
        self.last_output_text = ""
        self.silence_frames = 0
        self.cache = {}
        self.frame_count = 0
        self.audio_buffer = np.array([], dtype=np.float32)

        # 启动处理线程
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.daemon = True
        process_thread.start()

        # 启动录音流
        with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.2),  # 200ms块大小
                dtype=np.float32,
                device=device_index,
        ):
            print("🎤 麦克风已打开（静音检测模式）")
            print("💡 说话时自动识别，停顿超过0.5秒自动断句")
            while self.is_recording:
                time.sleep(0.1)

    def stop(self):
        """停止录音"""
        self.is_recording = False

        # 强制刷新剩余文本
        if self.accumulated_text:
            # 尝试清理末尾不完整内容
            clean_text = self.accumulated_text.strip()
            if clean_text and len(clean_text) >= 3:
                with result_lock:
                    recognition_results.append({
                        "time": time.strftime("%H:%M:%S"),
                        "text": clean_text
                    })
            self.accumulated_text = ""
            self.current_text = ""

    def get_current_text(self):
        """获取当前识别的文本（用于实时显示）"""
        return self.current_text

    def get_final_texts(self):
        """获取所有已归档的文本"""
        with result_lock:
            return recognition_results.copy()


# HTML 模板（与之前相同，为节省篇幅省略...）
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>实时语音识别 - 麦克风</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 800px;
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            background: rgba(255,255,255,0.2);
            font-size: 14px;
        }
        .status.active {
            background: #4CAF50;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .content {
            padding: 30px;
            min-height: 400px;
        }
        .current-text {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            min-height: 80px;
            font-size: 18px;
            line-height: 1.8;
            color: #333;
        }
        .current-text .label {
            font-size: 12px;
            color: #999;
            display: block;
            margin-bottom: 8px;
        }
        .history {
            max-height: 300px;
            overflow-y: auto;
        }
        .history-item {
            padding: 12px 15px;
            margin-bottom: 8px;
            background: #fafafa;
            border-left: 4px solid #667eea;
            border-radius: 5px;
            font-size: 16px;
            line-height: 1.6;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .history-time {
            color: #999;
            font-size: 12px;
            margin-right: 10px;
        }
        .controls {
            padding: 20px 30px;
            text-align: center;
            border-top: 1px solid #eee;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
            margin: 0 5px;
        }
        button:hover {
            transform: scale(1.05);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .clear-btn {
            background: #ff6b6b;
        }
        .device-select {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 10px;
            background: #f5f5f5;
        }
        .device-select select {
            padding: 8px 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-size: 14px;
            width: 100%;
            max-width: 400px;
        }
        .hint {
            font-size: 12px;
            color: #999;
            margin-top: 10px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 实时语音识别系统</h1>
            <div class="status" id="status">等待开始...</div>
        </div>
        <div class="content">
            <div class="device-select">
                <label>🎙️ 选择麦克风：</label>
                <select id="deviceSelect"></select>
            </div>
            <div class="current-text">
                <span class="label">🔊 正在识别：</span>
                <span id="currentText">等待识别结果...</span>
            </div>
            <h3>📝 识别历史（完整句子）：</h3>
            <div class="history" id="history"></div>
            <div class="hint">💡 提示：说话清晰，停顿自动断句</div>
        </div>
        <div class="controls">
            <button onclick="startRecognition()" id="startBtn">开始识别</button>
            <button onclick="stopRecognition()" id="stopBtn" disabled>停止识别</button>
            <button onclick="clearHistory()" class="clear-btn">清空历史</button>
        </div>
    </div>

    <script>
        let eventSource = null;
        let historyData = [];

        function loadDevices() {
            fetch('/api/devices')
                .then(response => response.json())
                .then(data => {
                    const select = document.getElementById('deviceSelect');
                    select.innerHTML = '';
                    data.devices.forEach(device => {
                        const option = document.createElement('option');
                        option.value = device.index;
                        option.textContent = device.name;
                        if (device.is_default) {
                            option.selected = true;
                        }
                        select.appendChild(option);
                    });
                });
        }

        function startRecognition() {
            const deviceIndex = document.getElementById('deviceSelect').value;
            fetch('/api/start', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ device_index: parseInt(deviceIndex) })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('status').className = 'status active';
                        document.getElementById('status').textContent = '🎤 正在录音识别...';
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                        document.getElementById('deviceSelect').disabled = true;
                        connectEventSource();
                    } else {
                        alert('启动失败: ' + data.message);
                    }
                });
        }

        function stopRecognition() {
            fetch('/api/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('status').className = 'status';
                        document.getElementById('status').textContent = '已停止';
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        document.getElementById('deviceSelect').disabled = false;
                        if (eventSource) {
                            eventSource.close();
                        }
                    }
                });
        }

        function clearHistory() {
            fetch('/api/clear', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    historyData = [];
                    document.getElementById('history').innerHTML = '';
                    document.getElementById('currentText').textContent = '等待识别结果...';
                });
        }

        function connectEventSource() {
            if (eventSource) {
                eventSource.close();
            }

            eventSource = new EventSource('/api/stream');

            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'current') {
                    document.getElementById('currentText').textContent = data.text || '正在识别...';
                } else if (data.type === 'new_result') {
                    addHistoryItem(data.result);
                }
            };

            eventSource.onerror = function() {
                console.log('EventSource 连接中断，尝试重连...');
                setTimeout(connectEventSource, 3000);
            };
        }

        function addHistoryItem(result) {
            historyData.push(result);
            const historyDiv = document.getElementById('history');
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `<span class="history-time">[${result.time}]</span>${result.text}`;
            historyDiv.appendChild(item);
            historyDiv.scrollTop = historyDiv.scrollHeight;
        }

        window.onload = function() {
            loadDevices();
            connectEventSource();
        };
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/devices')
def get_devices():
    devices = sd.query_devices()
    input_devices = []
    default_device = sd.default.device[0]

    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append({
                'index': i,
                'name': device['name'],
                'is_default': (i == default_device)
            })

    return jsonify({'devices': input_devices})


@app.route('/api/stream')
def stream():
    def generate():
        last_index = 0
        last_current = ""

        while True:
            time.sleep(0.15)  # 150ms轮询

            with result_lock:
                if len(recognition_results) > last_index:
                    for result in recognition_results[last_index:]:
                        yield f"data: {json.dumps({'type': 'new_result', 'result': result}, ensure_ascii=False)}\n\n"
                    last_index = len(recognition_results)

                if asr_engine:
                    current = asr_engine.get_current_text()
                    if current != last_current:
                        last_current = current
                        if current:
                            yield f"data: {json.dumps({'type': 'current', 'text': current}, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/start', methods=['POST'])
def start_recognition():
    global asr_engine

    if asr_engine and asr_engine.is_recording:
        return jsonify({"success": True, "message": "已经在识别中"})

    data = request.get_json() or {}
    device_index = data.get('device_index')

    if device_index is None:
        device_index = sd.default.device[0]

    try:
        device_info = sd.query_devices(device_index)
        if device_info['max_input_channels'] <= 0:
            return jsonify({"success": False, "message": "该设备不支持输入"})
    except Exception as e:
        return jsonify({"success": False, "message": f"设备无效: {str(e)}"})

    if asr_engine is None:
        asr_engine = RealTimeMicASR(sample_rate=16000, chunk_duration=1.5)

    threading.Thread(target=asr_engine.start, args=(device_index,), daemon=True).start()

    return jsonify({"success": True, "message": "识别已开始"})


@app.route('/api/stop', methods=['POST'])
def stop_recognition():
    global asr_engine

    if asr_engine:
        asr_engine.stop()

    return jsonify({"success": True, "message": "识别已停止"})


@app.route('/api/clear', methods=['POST'])
def clear_history():
    global recognition_results

    with result_lock:
        recognition_results.clear()

    if asr_engine:
        asr_engine.current_text = ""
        asr_engine.accumulated_text = ""
        asr_engine.cache = {}

    return jsonify({"success": True, "message": "历史已清空"})


def main():
    print("=" * 50)
    print("🎤 实时语音识别 Web 服务 (静音检测 + 智能断句)")
    print("=" * 50)
    print()

    print("可用的麦克风设备:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default_mark = " (默认)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default_mark}")

    print()
    print("📌 优化特性：")
    print("  - 静音检测断句，0.5秒静音自动切分")
    print("  - 智能断句，根据语义和标点输出完整句子")
    print("  - 实时增量显示，同时归档完整句子")
    print("  - 自动去重，避免输出重复内容")
    print()
    print("请在浏览器中打开: http://localhost:5000")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()