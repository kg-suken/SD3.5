import gradio as gr
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
from huggingface_hub import login
import os
from datetime import datetime

# トークンでログイン
login(token="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")

# モデルのIDと量子化済みモデルの保存パス
model_id = "stabilityai/stable-diffusion-3.5-large"
model_save_path = "quantized_model.pt"

# 出力ディレクトリの作成
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# 量子化済みモデルの読み込みまたは生成
if os.path.exists(model_save_path):
    model_nf4 = torch.load(model_save_path)
else:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )
    torch.save(model_nf4, model_save_path)

# パイプラインの設定
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

# 画像生成関数
def generate_image(
    prompt, height=1024, width=1024, num_inference_steps=50, 
    guidance_scale=7.0, negative_prompt=None, num_images_per_prompt=1
):
    images = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt
    ).images

    # 画像を保存
    saved_images = []
    for i, img in enumerate(images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png" if i == 0 else f"{timestamp}-{i+1}.png"
        img.save(os.path.join(output_dir, filename))
        saved_images.append(img)  # 保存した画像をリストに追加

    # 生成された画像が単一の場合でも、リストとして返す
    return saved_images if len(images) > 1 else [images[0]]

# Gradioインターフェースの定義
with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion 3.5 Image Generator")

    with gr.Row():
        # 左側の入力エリア
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", placeholder="Describe the image you want to generate")
            height = gr.Slider(label="Height", minimum=256, maximum=4096, value=1024, step=16)
            width = gr.Slider(label="Width", minimum=256, maximum=4096, value=1024, step=16)
            num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100, value=50)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=7.0)
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompts if any")
            num_images_per_prompt = gr.Slider(label="Number of Images per Prompt", minimum=1, maximum=5, step=1, value=1)
            generate_button = gr.Button("Generate")

        # 右側の出力エリア
        with gr.Column(scale=1):
            output = gr.Gallery(label="Generated Images")

    # クリックイベントの設定
    generate_button.click(
        fn=generate_image,
        inputs=[prompt, height, width, num_inference_steps, guidance_scale, negative_prompt, num_images_per_prompt],
        outputs=output
    )

demo.launch()
