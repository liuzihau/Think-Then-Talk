python eval_t3.py \
  --model t3_model \
  --model_args ckpt_path=cfg5-8-32-2/state_2,gen_length=256,block_size=8,think_device1=cuda:0,think_device2=cuda:0,talk_device=cuda:0,show_speed=True \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 1 \
  --output_path ./lm_eval_outputs
