import gc
import urllib
from enum import Enum

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from basicsr.utils.download_util import load_file_from_url
import numpy as np
import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from typing import Optional
from fastapi.responses import StreamingResponse
import io
import torch
import threading

app = FastAPI()

# 创建线程锁
lock = threading.Lock()


# 枚举模型可选项
class ModelName(str, Enum):
    RealESRGAN_x4plus = "RealESRGAN_x4plus"
    RealESRNet_x4plus = "RealESRNet_x4plus"
    RealESRGAN_x4plus_anime_6B = "RealESRGAN_x4plus_anime_6B"
    RealESRGAN_x2plus = "RealESRGAN_x2plus"
    realesr_animevideov3 = "realesr-animevideov3"
    realesr_general_x4v3 = "realesr-general-x4v3"


# 图像处理函数
def process_image(
        img, model_name, denoise_strength, outscale, tile, tile_pad, pre_pad,
        face_enhance, fp32, alpha_upsampler, gpu_id
):
    with lock:  # 使用线程锁来防止并发处理
        # 选择模型并设置模型路径
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRNet_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif model_name == 'realesr-animevideov3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif model_name == 'realesr-general-x4v3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        # 确定模型路径
        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        # 控制去噪强度
        dni_weight = None
        if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]

        # 初始化放大器
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32 and torch.cuda.is_available(),  # 只有在 GPU 上才使用半精度
            gpu_id=gpu_id,
        )

        # 可选人脸增强
        if face_enhance:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler
            )

        try:
            if face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=outscale, alpha_upsampler=alpha_upsampler)
        except RuntimeError as error:
            raise HTTPException(status_code=500, detail=f"请求发送错误: {error}")
        # 显式删除变量并清空缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        # 返回处理后的图像
        return output


@app.post("/upscale")
async def upscale_image(
        file: UploadFile = File(...),
        model_name: ModelName = Form(default="RealESRGAN_x4plus"),
        denoise_strength: float = Form(default=0.5, ge=0.0, le=1.0),
        outscale: float = Form(default=4),
        tile: int = Form(default=0),
        tile_pad: int = Form(default=10),
        pre_pad: int = Form(default=0),
        face_enhance: bool = Form(default=False),
        fp32: bool = Form(default=False),
        alpha_upsampler: str = Form(default='realesrgan'),
        gpu_id: Optional[int] = Form(default=0),
):
    # 读取上传的图像文件
    contents = await file.read()
    img_name = file.filename  # 获取上传文件的原始名称
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise HTTPException(status_code=400, detail="无法读取上传的图像文件，请检查文件格式")

    try:
        # 使用 run_in_threadpool 来异步执行图像处理任务
        output = await run_in_threadpool(
            process_image,
            img=img,
            model_name=model_name,
            denoise_strength=denoise_strength,
            outscale=outscale,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            face_enhance=face_enhance,
            fp32=fp32,
            alpha_upsampler=alpha_upsampler,
            gpu_id=gpu_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # 将输出图像编码为字节流格式
    _, img_encoded = cv2.imencode('.png', output)
    img_bytes = io.BytesIO(img_encoded)

    # 创建响应并设置文件名
    response = StreamingResponse(
        img_bytes,
        media_type="image/png"
    )
    encoded_img_name = urllib.parse.quote(os.path.basename(img_name))
    response.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{encoded_img_name}"

    # 返回图像流作为响应
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
