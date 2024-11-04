
# StableDiffusion3.5をPythonで動かす方法

間違いなどがあったら指摘していただけるとうれしいです。質問もお待ちしています。

## Nvidia関係のコマンド

```
nvidia-smi
```

```
nvcc -V
```

## コマンド

1:
```
python -m venv venv
```

2:
```
.\venv\Scripts\activate
```

3:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

4:
```
pip install -U diffusers accelerate protobuf transformers bitsandbytes huggingface sentencepiece
```

## プログラム

[Download app.py](./app.py)

## Gradio(おまけ)

```
pip install gradio
```

[Gradio-Download app-gra.py](./app-gra.py)

## リンクなど

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
