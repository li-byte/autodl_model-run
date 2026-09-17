import sounddevice as sd
import numpy as np
import queue
import threading
from funasr import AutoModel
import time
import sys

# 清除代理设置
import os

for key in list(os.environ.keys()):
    if "proxy" in key.lower():
        os.environ.pop(key, None)


class RealTimeSystemAudioASR:
    def __init__(self, sample_rate=16000, chunk_duration=0.6):
        """
        初始化实时系统音频识别

        Args:
            sample_rate: 采样率
            chunk_duration: 每个音频块时长（秒）
        """
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()

        print("正在加载模型...")
        self.model = AutoModel(
            model="./models/paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            device="cuda:0",  # 使用 GPU
        )
        print("模型加载完成！")

        self.is_recording = False
        self.cache = {}

    def audio_callback(self, indata, frames, time_info, status):
        """音频回调函数"""
        if status:
            print(f"状态: {status}", file=sys.stderr)

        # 将音频数据放入队列（如果是立体声，取平均转为单声道）
        if indata.shape[1] > 1:
            audio_chunk = np.mean(indata, axis=1).astype(np.float32)
        else:
            audio_chunk = indata[:, 0].copy().astype(np.float32)

        self.audio_queue.put(audio_chunk)

    def process_audio(self):
        """处理音频队列"""
        audio_buffer = np.array([], dtype=np.float32)

        while self.is_recording:
            try:
                # 获取音频块
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, chunk])

                # 当积累足够的音频时处理
                if len(audio_buffer) >= self.chunk_samples:
                    # 取一个 chunk 的音频
                    process_chunk = audio_buffer[:self.chunk_samples]
                    audio_buffer = audio_buffer[self.chunk_samples:]

                    # 调用 FunASR 流式识别
                    result = self.model.generate(
                        input=process_chunk,
                        cache=self.cache,
                        is_final=False,
                        chunk_size=[0, 10, 5],
                    )

                    # 输出识别结果
                    if result and len(result) > 0:
                        text = result[0].get("text", "")
                        if text:
                            print(f"\r🎤 {text}", end="", flush=True)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n处理错误: {e}")

    def start(self, device_index=None):
        """开始实时识别"""
        print("=" * 50)
        print("实时系统音频识别已启动")
        print("按 Ctrl+C 停止")
        print("=" * 50)

        self.is_recording = True

        # 启动音频处理线程
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.daemon = True
        process_thread.start()

        try:
            # 打开音频流（使用 WASAPI 环回设备）
            with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=2,  # 系统音频通常是立体声
                    callback=self.audio_callback,
                    blocksize=self.chunk_samples,
                    dtype=np.float32,
                    device=device_index,  # 使用系统音频环回设备
            ):
                print("🔊 系统音频流已打开，开始识别...")

                # 保持运行
                while self.is_recording:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n停止识别...")
        except Exception as e:
            print(f"\n错误: {e}")
        finally:
            self.is_recording = False
            print("已停止")

    def stop(self):
        """停止识别"""
        self.is_recording = False


def list_loopback_devices():
    """列出所有音频设备（包括系统音频环回设备）"""
    print("=" * 50)
    print("可用的音频设备:")
    print("=" * 50)

    devices = sd.query_devices()
    loopback_devices = []

    for i, device in enumerate(devices):
        # 寻找环回设备或输出设备
        if device['max_input_channels'] > 0:
            device_type = "输入"
            if "loopback" in device['name'].lower() or "stereo mix" in device['name'].lower() or "立体声混音" in device[
                'name']:
                device_type = "🔊 系统音频环回"
                loopback_devices.append(i)

            print(f"[{i}] {device['name']} - 类型: {device_type} - 输入通道: {device['max_input_channels']}")

        # 也显示输出设备（某些系统需要特殊配置）
        elif device['max_output_channels'] > 0:
            print(f"[{i}] {device['name']} - 类型: 输出 - 输出通道: {device['max_output_channels']}")

    print("=" * 50)

    if loopback_devices:
        print(f"找到 {len(loopback_devices)} 个系统音频环回设备")
    else:
        print("⚠️  未找到明显的环回设备，请检查系统设置")
        print("提示：Windows 需要启用'立体声混音'或使用虚拟音频设备")

    return devices


def get_system_audio_device():
    """获取系统音频环回设备"""
    devices = sd.query_devices()

    # 尝试找到环回设备
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if device['max_input_channels'] > 0:
            # Windows: 查找 "立体声混音" 或 "Stereo Mix"
            if "stereo mix" in name or "立体声混音" in name or "loopback" in name:
                return i

    # 如果没有找到专门的环回设备，尝试使用 WASAPI
    try:
        import soundcard as sc
        speakers = sc.all_speakers()
        if speakers:
            return speakers[0]
    except:
        pass

    return None


def main():
    """主函数"""
    # 1. 列出音频设备
    devices = list_loopback_devices()

    # 2. 获取系统音频设备
    device_index = get_system_audio_device()

    if device_index is not None:
        print(f"\n✅ 使用系统音频设备: [{device_index}] {devices[device_index]['name']}")
    else:
        print("\n⚠️  未找到系统音频环回设备")
        print("请尝试以下方法：")
        print("1. Windows: 右键音量图标 → 声音 → 录音 → 启用'立体声混音'")
        print("2. 安装虚拟音频设备，如 VB-Cable")
        print("3. 手动指定设备索引")

        # 让用户手动输入设备索引
        try:
            manual_index = int(input("\n请输入要使用的设备索引（从上面的列表中）: "))
            if 0 <= manual_index < len(devices):
                device_index = manual_index
            else:
                print("无效的设备索引")
                return
        except ValueError:
            print("无效输入")
            return

    # 3. 创建实时识别实例
    asr = RealTimeSystemAudioASR(
        sample_rate=16000,
        chunk_duration=0.6,  # 600ms 一个块
    )

    # 4. 开始实时识别
    try:
        asr.start(device_index=device_index)
    except KeyboardInterrupt:
        print("\n程序退出")
    finally:
        asr.stop()


if __name__ == "__main__":
    main()