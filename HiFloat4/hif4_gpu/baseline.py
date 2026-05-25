import torch
import torch.nn as nn
from diffusers import WanImageToVideoPipeline
from diffusers.utils import load_image
from diffusers.utils import export_to_video

# 引入 GCC 官方提供的 HiFloat4 GPU 算子
from quant_cy import QType, quant_dequant_float

# ==========================================
# 1. 定义 HiF4 量化线性层 (融合 SmoothQuant)
# ==========================================
class HiF4Linear(nn.Module):
    def __init__(self, original_linear, smooth_scale=None, act_quant_dim=-1):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # 克隆原始权重和偏置
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
            
        # 官方 HiF4 QType 配置
        self.qtype_str = 'hifx4'
        self.weight_quant_type = QType(self.qtype_str).dim(0)           # Per-Channel 权重
        self.act_quant_type = QType(self.qtype_str).dim(act_quant_dim)  # Per-Token 激活
        
        # 融合 SmoothQuant 平滑因子 (将激活异常值转移到权重上)
        if smooth_scale is not None:
            self.register_buffer('smooth_scale', smooth_scale)
            self.weight.data = self.weight.data * self.smooth_scale.view(1, -1)
        else:
            self.smooth_scale = None

    def forward(self, x):
        # 激活值平滑
        if self.smooth_scale is not None:
            x = x / self.smooth_scale.view(1, -1)
            
        # 动态伪量化：每次 forward 都会根据当前 timestep 的激活值动态计算 scale
        x_sim = quant_dequant_float(x, self.act_quant_type, force_py=False, force_fp32=True)
        w_sim = quant_dequant_float(self.weight, self.weight_quant_type, force_py=False, force_fp32=True)
        
        return nn.functional.linear(x_sim, w_sim, self.bias)

# ==========================================
# 2. 递归替换模型层 (精准控制混合精度)
# ==========================================
def replace_linear_with_hif4(module, sensitive_layers, smooth_scales_dict, current_path=""):
    replaced_count = 0
    skipped_count = 0
    
    for name, child in module.named_children():
        full_name = f"{current_path}.{name}" if current_path else name
        
        if isinstance(child, nn.Linear):
            # 检查当前层是否在敏感层白名单中
            is_sensitive = any(full_name.endswith(s_layer) for s_layer in sensitive_layers)
            
            if is_sensitive:
                print(f"[Mixed Precision] 🛡️ 保留全精度层: {full_name}")
                skipped_count += 1
            else:
                # 获取校准好的 scale 并替换为 HiF4 层
                scale = smooth_scales_dict.get(full_name, None)
                hif4_layer = HiF4Linear(child, smooth_scale=scale)
                setattr(module, name, hif4_layer)
                replaced_count += 1
        else:
            # 递归遍历子模块 (如 transformer.blocks)
            r, s = replace_linear_with_hif4(child, sensitive_layers, smooth_scales_dict, full_name)
            replaced_count += r
            skipped_count += s
            
    return replaced_count, skipped_count

# ==========================================
# 3. 主程序：加载、量化与推理
# ==========================================
def main():
    print("🚀 1. 加载 Wan2.2-I2V-A14B 模型...")
    # 推荐使用 Diffusers 格式加载，它是目前 HF 生态最兼容的格式
    model_id = "/mnt/diskhd/Backup/DownloadModel/Wan2.2-I2V-A14B-BF16/" 
    
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    
    print("📊 2. 准备 SmoothQuant Scales...")
    # 实际比赛中，你需要用 10-20 个视频跑一遍全精度前向传播，统计 max(abs(X))
    # 这里为了演示直接传入空字典，HiF4 原生的动态量化也能扛住大部分误差
    smooth_scales_dict = {} 
    
    print("⚙️ 3. 应用 HiF4 量化与混合精度策略...")
    # 【冠军策略】：赛题规定 HiF4 最多保留 2 层全精度。
    # 对于 Wan2.2 的 DiT 结构，最敏感的 2 层通常是：
    # 1. proj_out: 最后的输出映射，直接决定生成视频的 Latent 噪点，量化极易导致画面全损。
    # 2. time_text_embed.timestep_embedder.linear_2: 时间步嵌入的核心层，决定了去噪的节奏。
    SENSITIVE_LAYERS_HIF4 =[
        "proj_out", 
        "time_text_embed.timestep_embedder.linear_2"
    ]
    
    # 我们只对 DiT Transformer 进行量化，VAE 和 Text Encoder 保持原样
    replaced, skipped = replace_linear_with_hif4(
        pipe.transformer, 
        sensitive_layers=SENSITIVE_LAYERS_HIF4, 
        smooth_scales_dict=smooth_scales_dict
    )
    print(f"✅ 替换完成！成功量化层数: {replaced}, 保留全精度层数: {skipped} (必须 <= 2)")
    
    print("🎬 4. 运行 HiF4 量化后的图生视频推理...")
    # 加载一张测试图片
    image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG")
    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
    
    # 开启显存优化 (Wan2.2 14B 即使量化后也需要一定显存)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    
    # 生成视频
    output = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=50,
        guidance_scale=7.0
    )
    
    video_frames = output.frames[0]
    export_to_video(video_frames, "wan2_2_hif4_output.mp4", fps=24)
    print("🎉 推理完成！视频已保存为 wan2_2_hif4_output.mp4")

if __name__ == "__main__":
    main()