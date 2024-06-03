# Transformer_from_scratch
AIDL project: ASR whisper finetuning + RAG    


whisper-base 로 전처리 한 데이터 : allenpoe/atcosim_prepared_whisper_base
whisper-base 로 finetune 한 모델    
model_address = 'allenpoe/whisper-base-atco'     

1000 epoch, validate every 100 epoch : 2시간 정도
validation도 시간이 걸리므로 200번마다 혹은 더 길게 해도 될듯     

Step	Training Loss	Validation Loss	Wer
100	1.657200	1.269409	71.009628
200	0.337300	0.322521	16.429644
300	0.147000	0.165291	9.212967
400	0.097500	0.110657	6.333363
500	0.054400	0.089244	6.258281
600	0.034400	0.076036	5.410299
700	0.051700	0.066345	5.039307
800	0.039600	0.060731	4.310573
900	0.029700	0.056570	4.270824
1000	0.013300	0.055709	4.169243     

whisper-base 로 그냥 evaluate
{'eval_loss': 3.243475914001465,
 'eval_wer': 99.0990195212437,
 'eval_runtime': 649.8269,
 'eval_samples_per_second': 2.925,
 'eval_steps_per_second': 0.366}    

 https://colab.research.google.com/drive/1OYrBOir_fELBskVcFo9YpTg3gpniiorE?usp=sharing