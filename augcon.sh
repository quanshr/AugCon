nohup python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen1.5-32B-Chat \
     --tensor-parallel-size 8 echo $! > pid.txt
python step1_gen_q.py
kill $(cat pid.txt)

deepspeed step2_scorer.py

nohup python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen1.5-32B-Chat \
     --tensor-parallel-size 8 & echo $! > pid.txt
python step3_gen_a.py
kill $(cat pid.txt)