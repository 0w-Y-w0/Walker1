# prepare_corpus.py
# 4.4.1 准备语料
# 用途: 准备 input.txt（tiny Shakespeare 英文小语料, 约 1MB, 1115394 字符）,
#       供后续字符级语言模型训练/量化部署实验使用。也可换成中文小语料。
# 方式1: 用现成 tiny shakespeare（需联网下载一次, 之后本地缓存复用）
# 常见报错与解决:
#   - raw.githubusercontent.com 国内直连不稳定 -> 备用镜像 jsdelivr;
#   - 两者都失败 -> 手动下载放到本目录并命名为 input.txt。
import urllib.request
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
mirror = "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt"

if os.path.exists("input.txt"):
    print("input.txt 已存在, 跳过下载。")
else:
    try:
        urllib.request.urlretrieve(url, "input.txt")
        print("已下载 tiny shakespeare 到 input.txt")
    except Exception as e:
        print("官方源下载失败, 尝试 jsdelivr 镜像:", e)
        try:
            urllib.request.urlretrieve(mirror, "input.txt")
            print("已通过镜像下载 tiny shakespeare 到 input.txt")
        except Exception as e2:
            print("下载失败，请手动准备 input.txt:", e2)

if os.path.exists("input.txt"):
    text = open("input.txt", encoding="utf-8").read()
    print(f"语料规模: {len(text)} 字符, 前 80 字符预览:")
    print(repr(text[:80]))
