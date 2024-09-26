# FAQ

1. **Q: How to select models?**<br>
A: Please refer to [docs/model_zoo.md](docs/model_zoo.md)

1. **Q: Can `face_enhance` be used for anime images/animation videos?**<br>
A: No, it can only be used for real faces. It is recommended not to use this option for anime images/animation videos to save GPU memory.

1. **Q: Error "slow_conv2d_cpu" not implemented for 'Half'**<br>
A: In order to save GPU memory consumption and speed up inference, Real-ESRGAN uses half precision (fp16) during inference by default. However, some operators for half inference are not implemented in CPU mode. You need to add **`--fp32` option** for the commands. For example, `python inference_realesrgan.py -n RealESRGAN_x4plus.pth -i inputs --fp32`.

1. **Q: windows basicsr安装失败报错INFO: pip is looking at multiple versions of basicsr to determine which version is compatible with other requirements. This could take a while.
ERROR: Could not find a version that satisfies the requirement tb-nightly (from basicsr) (from versions: none)
ERROR: No matching distribution found for tb-nightly
?**<br>
A:pip install -i https://mirrors.aliyun.com/pypi/simple tb-nightly
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple basicsr==1.4.2
    