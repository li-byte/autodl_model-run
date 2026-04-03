import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# -----------------------
# 配置参数
# -----------------------
model_id = "../../models/wan2_1-1_3b"  # WAN2.1 VACE 模型路径
height, width = 480, 816               # 视频分辨率
fps = 16                               # 帧率
total_seconds = 12                      # 视频时长（秒）
frames_per_segment = 16 * 1             # 每段帧数（分段生成防 OOM）
guidance_scale = 5.0                     # 采样引导强度

if __name__ == '__main__':
    # 中文 Prompt
    prompt = (
        "宁静的秋日下午，阳光透过稀疏的云层洒在林间小径上，"
        "枫树的叶子呈红色、橙色和金黄色，微风吹过时几片枫叶轻轻飘落，旋转着落在石板路上。"
        "空气中弥漫着泥土和落叶的清香，林间景色温暖柔和，光线自然真实，充满秋天的氛围。"
        "镜头运动：慢速推进，温和的电影平移，沿林间缓慢前行的跟拍镜头，"
        "枫叶飘落的特写镜头，浅景深，慢动作，电影感，柔和光线，高质量，画面稳定"
    )
    negative_prompt = (
        "人物、动物、过曝、模糊、低质量、不自然的颜色、JPEG 压缩痕迹、"
        "额外肢体、手或脸画得不好、透视变形、字幕、文字、卡通风格、绘画风格、"
        "杂乱背景、不自然光线、物体变形"
    )

    # -----------------------
    # 计算总帧数
    # -----------------------
    total_frames = total_seconds * fps

    # -----------------------
    # 加载 VAE
    # -----------------------
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )

    # -----------------------
    # 配置光流调度器
    # -----------------------
    flow_shift = 5.0
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift
    )

    # -----------------------
    # 加载 WAN VACE Pipeline
    # -----------------------
    pipe = WanVACEPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16
    )
    pipe.scheduler = scheduler
    pipe.to("cuda")

    # -----------------------
    # 分段生成视频
    # -----------------------
    all_frames = []
    prev_latents = None

    for i in range(0, total_frames, frames_per_segment):
        current_num_frames = min(frames_per_segment, total_frames - i)


        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=current_num_frames,
                guidance_scale=guidance_scale,
                latents=prev_latents  # 传递上一段最后一帧的 latent
            )

        segment_frames = out.frames
        all_frames.extend(segment_frames[0])

        # 更新 prev_latents 为当前段最后一帧 latent
        prev_latents = out.latents[:, -1:]

        # 清理显存
        del out
        torch.cuda.empty_cache()

        print(f"段 {i // frames_per_segment + 1} 完成, 当前总帧数: {len(all_frames)}")

    # -----------------------
    # 导出视频
    # -----------------------
    export_to_video(all_frames, "continuous_forest.mp4", fps=fps)
    print("视频生成完成: continuous_forest.mp4")