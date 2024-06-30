from peft import LoraConfig

# model_name_or_path = "/home/work/workspace/models/Qwen1_8-Chat"
# model_name_or_path = "/home/work/workspace/models/Qwen1___5-32B-Chat"
# model_name_or_path = "/home/work/workspace/models/Llama-2-7b-chat-ms"
# model_name_or_path = "/home/work/workspace/models/Llama-2-13b-chat-ms"
# model_name_or_path = "/home/work/workspace/models/Meta-Llama-3-8B-Instruct"
# model_name_or_path = "/home/work/workspace/models/Meta-Llama-3-70B-Instruct"


model_name_or_path = "Qwen/Qwen1.5-32B-Chat"


lora_config = LoraConfig(
            r=32, 
            lora_alpha=32, 
            target_modules="all-linear", # for QLoRA
            lora_dropout=0.05, 
            bias="none",
        )

# dataset_path = '/home/work/workspace/project3/datasets/drop'
# dataset_path = '/home/work/workspace/project3/datasets/squad'
# dataset_path = '/home/work/workspace/project3/datasets/triviaqa'
# dataset_path = '/home/work/workspace/project3/datasets/webglm-qa'

max_length = 500  # 问题的最大粒度（从长度为500的文本中提取问题）
min_length = 20   # 问题的最小粒度

character_per_piece = 35 # 每35个汉字提取一组数据

rouge_thres = 0.7 # 判断相似度的阈值

q_temperature = 0.85 # 生成问题的温度
a_temperature = 0.2  # 生成回复的温度