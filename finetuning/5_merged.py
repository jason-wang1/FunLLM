import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, TaskType

# 模型和权重的路径
model_path = './LLM-Research/Meta-Llama-3-8B-Instruct'
lora_path = './llama3_lora'

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载原始大模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

# 配置 Lora 参数
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 如果是进行推理，则设置为True
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alpha
    lora_dropout=0.1  # Dropout 比例
)

# 加载 Lora 微调后的权重到模型
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

# 现在模型已经整合了微调后的权重，我们可以将其保存为一个新的模型
model.save_pretrained('./Huanhuan-Llama3-Model')

# 你也可以保存 tokenizer 如果需要
tokenizer.save_pretrained('./Huanhuan-Llama3-Model')
