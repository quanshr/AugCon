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

max_length = 500  # Maximum granularity of the problem
min_length = 20   # Minimum granularity of the problem

character_per_piece = 35 # Extract a set of data for every 35 characters

rouge_thres = 0.7 # Threshold for determining similarity

q_temperature = 0.85 # Temperature for generating questions
a_temperature = 0.2  # Temperature for generating responses