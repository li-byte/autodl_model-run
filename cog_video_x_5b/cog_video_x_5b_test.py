import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

if __name__ == '__main__':
    # 限制当前进程最多使用 90% 的 GPU 显存
    torch.cuda.set_per_process_memory_fraction(0.9, device=0)
    prompt = """
    Autumn maple forest, red and golden leaves slowly falling, gentle wind moving tree branches, ground covered with fallen leaves, small flowing stream, light mist, warm sunset light filtering through trees, orange sky

    camera movement: slow dolly in, gentle cinematic pan, slow forward tracking shot through the forest, close-up of falling maple leaves, shallow depth of field, slow motion, cinematic, soft lighting, high quality, stable
        """


    pipe = CogVideoXPipeline.from_pretrained(
        "../../models/cog_video_x_5b",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    export_to_video(video, "output.mp4", fps=8)
