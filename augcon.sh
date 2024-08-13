nohup python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen1.5-32B-Chat \
     --tensor-parallel-size 8 > vllm.log 2>&1 &
sleep 3m
python step1_gen_q.py
pkill -9 -e -f vllm

deepspeed step2_scorer.py

nohup python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen1.5-32B-Chat \
     --tensor-parallel-size 8 > vllm.log 2>&1 &
sleep 3m
python step3_gen_a.py
pkill -9 -e -f vllm