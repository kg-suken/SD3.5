from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
from huggingface_hub import login
import os

# トークンでログイン
login(token="hf_")

model_id = "stabilityai/stable-diffusion-3.5-large"
model_save_path = "quantized_model.pt"  # 量子化済みモデルの保存パス

if os.path.exists(model_save_path):
    # 量子化済みモデルが保存されている場合、読み込む
    model_nf4 = torch.load(model_save_path)
else:
    # 量子化設定
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # モデルを量子化して読み込む
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )
    # 量子化済みモデルを保存
    torch.save(model_nf4, model_save_path)

# パイプラインの設定
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

# プロンプトを使って画像生成
prompt = "An anime girl holoding a sign says Hello World"
image = pipeline(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=4.5,
    max_sequence_length=512,
).images[0]
image.save("SD3.5.png")
