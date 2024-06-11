# Transformer_from_scratch
AIDL project: ASR whisper finetuning + RAG    

DATASET : Jzuluaga/atcosim_corpus       
변환된 DATASET : allenpoe/atcosim_dataset_for_finetune_whisper_small     

ATCOSIM으로 FINTUNE 된 WHISPER SMALL 모델 : allenpoe/whisper-small-finetune-atcosim-final    

----------------------------------


파인튜닝    
 https://colab.research.google.com/drive/1OYrBOir_fELBskVcFo9YpTg3gpniiorE?usp=sharing    

여러 모델로 추론 + 간단한 RAG    
 https://colab.research.google.com/drive/13mRwBzHwsHtSi3FzSv2oNISKpKOuf-Dw?usp=sharing

 converted 는 숫자 표기법 123 식으로 변환한 것

## Base   

### Base original

whisper-base 로 전처리 한 데이터 : allenpoe/atcosim_prepared_whisper_base    
whisper-base 로 finetune 한 모델 : allenpoe/whisper-base-atco        

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

 ### Base converted

whisper-base 로 전처리 한 데이터 : allenpoe/atcosim_converted_prepared_whisper_base    

 Step	Training Loss	Validation Loss	Wer    
100	1.882000	1.529038	67.414301    
200	0.433400	0.425137	26.258704    
300	0.188300	0.221765	15.593198    
400	0.125900	0.148661	10.986877    
500	0.071300	0.116831	9.413498    
600	0.044600	0.103563	7.565613    
700	0.065200	0.091920	6.320300    
800	0.050800	0.082914	5.115158    

 

 ## Small   

 ### Small converted

whisper-small 로 전처리 한 데이터 : allenpoe/atcosim_converted_prepared_whisper_small    
whisper-small 로 finetune 한 모델 : allenpoe/whisper-small-atco-converted

Step	Training Loss	Validation Loss	Wer    
100	1.603300	1.360189	78.106588    
200	0.616100	0.599254	21.792983    
300	0.110300	0.127348	14.736208    
400	0.063100	0.086483	13.758704    
500	0.036900	0.076273	11.964381    
600	0.019800	0.064437	16.811730    
700	0.036900	0.059947	10.437868    
800	0.027200	0.051758	12.192019    
900	0.020200	0.047269	8.415908    
1000	0.006400	0.046787	7.204071     
    
whisper-small 로 그냥 evaluate 
{'eval_loss': 4.3919677734375,     
 'eval_wer': 84.68800214247456,    
 'eval_runtime': 886.7614,    
 'eval_samples_per_second': 2.144,    
 'eval_steps_per_second': 0.268}    

------------------------

whisper small finetune 모델에 ATCO2 데이터     
https://colab.research.google.com/drive/1nfO7-o7QH-X_99dt0yClIDOqYlFqV6X-?usp=sharing     


whisper finetune llama2     
https://colab.research.google.com/drive/1VxpNomvVx2LMGYiLAk4USu1wXaY4UeyR?usp=sharing    

whisper finetune llama2 허깅페이스에서 모델 불러오기     
https://colab.research.google.com/drive/1kqAasjPiHB8o7CEuvFjMMg4HBdQm9tQ7?usp=sharing
 
 
-------------------------
