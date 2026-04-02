import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

if __name__ == '__main__':

    prompt = "一只熊猫，穿着一件红色的小夹克，戴着一顶小帽子，坐在宁静的竹林里的木凳上。熊猫毛茸茸的爪子拨弄着一把微型原声吉他，发出柔和的旋律。附近，其他几只熊猫聚集在一起，好奇地看着，有些还有节奏地鼓掌。阳光透过高大的竹子，在现场投下柔和的光芒。熊猫的脸很有表情，在玩耍时表现出专注和快乐。背景包括一条小溪和生机勃勃的绿叶，增强了这场独特音乐表演的宁静和神奇氛围。"


    pipe = CogVideoXPipeline.from_pretrained(
        "../../cog_video_x_5b",
        torch_dtype=torch.bfloat16
    ).to("cuda")  # 5090 不需要 cpu offload

    pipe.vae.enable_tiling()  # 保留即可

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=60,  # ↑ 提高质量
        num_frames=81,  # ↑ 更长视频
        guidance_scale=7,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    export_to_video(video, "output.mp4", fps=8)
