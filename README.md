# Transformer_from_scratch
AIDL project: ASR whisper finetuning + RAG    

--------------------------------
# ATCO2_test 부분은 여기다 작성 부탁드려요 (비어있는거 채우기)           

DATASET : Jzuluaga/atco2_corpus_1h        
     
WHISPER SMALL : openai/whisper-small     
      
WHISPER FINETUNED : allenpoe/whisper-small-finetuned-atcosim-model-final  
       
LLAMA2 : NousResearch/Llama-2-7b-chat-hf  이거 맞나요?     
meta-llama/Llama-2-7b-chat-hf 이게 메타에서 나온건데 이건 허깅페이스 계정 승인 받고 로그인 후 사용 가능합니다.         
승인받은 계정       
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      
      
LLAMA2 FINETUNED : KooJM/llama-2-7b_finetuned_using_whisper-small

### 1. 위스퍼기본    

##### 코랩     
https://colab.research.google.com/drive/1Q5ZUxbHooEdYtJvhu_Su1XhfmcnpXSOi?usp=sharing      

##### 결과      
WER 평균 :  1.0405270031874394       
allenpoe/atco2_test_dictation_by_whisper_small     
atco2_test_dictation_by_whisper_small.csv     

### 2. 위스퍼파인튜닝       

##### 코랩
https://colab.research.google.com/drive/1cxmbCTmX0UdFPyFRkcpy-ULiI5eHRJbY?usp=sharing    

##### 결과
WER 평균 :  1.0495432499523096    
allenpoe/atco2_test_dictation_by_whisper_finetuned        
atco2_test_dictation_by_whisper_finetuned.csv     

### 3. 위스퍼기본 + 라마기본

##### 코랩     

##### 결과      


### 4. 위스퍼기본 + 라마파인튜닝

##### 코랩        
코랩만 작성하고 시간부족으로 못돌렸습니다.    

https://colab.research.google.com/drive/1gwUnGpV46C1grnj4r30Icp9kVCxczTJj?usp=sharing    

https://colab.research.google.com/drive/15zEvy6hq3Zu6fJ5X2Wg13UIOCKxnD5FF?hl=ko#scrollTo=NIUpvoTd3FDW
##### 결과      
WER 뽑는 중
라마 파인튜닝 모델 (KooJM/llama-2-7b_finetuned_using_whisper-small)

atco2_test_dictation_by_whisper_small.csv
### 5. 위스퍼파인튜닝 + 라마기본

##### 코랩     

##### 결과      


### 6. 위스퍼파인튜닝 + 라마 파인튜닝

##### 코랩     

##### 결과      




----------------------------------

DATASET : Jzuluaga/atcosim_corpus       
변환된 DATASET : allenpoe/atcosim_dataset_for_finetune_whisper_small     

ATCOSIM으로 FINTUNE 된 WHISPER SMALL 모델 : allenpoe/whisper-small-finetuned-atcosim-model-final    

파인튜닝된 모델로 atcosim 받아쓰기 데이터셋 (train, test 둘 다 한 주소에 올렸습니다. wav파일은 이름만 있고 오디오파일이 아닙니다. 텍스트만 있어요) :        
allenpoe/ATCOSIM_dictation_by_finetuned_whisper_small     

파인튜닝안된 모델로 atcosim 받아쓰기 데이터셋 (숫자 표기 변환X)
KooJM/Whisper-small_ACTOSIM_Train_data
KooJM/Whisper-small_ACTOSIM_Test_data


----------------------------------


파인튜닝    
 https://colab.research.google.com/drive/1OYrBOir_fELBskVcFo9YpTg3gpniiorE?usp=sharing    

여러 모델로 추론 + 간단한 RAG    
 https://colab.research.google.com/drive/13mRwBzHwsHtSi3FzSv2oNISKpKOuf-Dw?usp=sharing

 converted 는 숫자 표기법 123 식으로 변환한 것


라마 + RAG. 위스퍼는 안붙은 상태. 앞부분의 간단 버전만 해도 될 듯 합니다.    
https://colab.research.google.com/drive/1h0WrDfO6fyZ-vyHrtTciLMYiWhze5nSX?usp=sharing

------------------------

whisper small finetune 모델에 ATCO2 데이터     
https://colab.research.google.com/drive/1nfO7-o7QH-X_99dt0yClIDOqYlFqV6X-?usp=sharing     


whisper finetune llama2     
https://colab.research.google.com/drive/1VxpNomvVx2LMGYiLAk4USu1wXaY4UeyR?usp=sharing    

whisper finetune llama2 허깅페이스에서 모델 불러오기     
https://colab.research.google.com/drive/1kqAasjPiHB8o7CEuvFjMMg4HBdQm9tQ7?usp=sharing

Llama_2 7b 파인튜닝하기 
  - https://colab.research.google.com/drive/1g7Eyjy9tPMY77B6wBtV9DU2T26eMQ16L?usp=sharing
  - step1 : Whisper-small에 ATCOSIM train과 test의 "audio" 집어넣어서 받아쓰기 생성
          ※ Whisper-small에 ATCOSIM train 받아쓰기 데이터 : "KooJM/Whisper-small_ACTOSIM_Train_data"
          ※ Whisper-small에 ATCOSIM test 받아쓰기 데이터 : "KooJM/Whisper-small_ACTOSIM_Test_data"
  - step2 : Llama_2 7b에 받아쓰기한걸 prompt로 넣어주고, 정답은 ATCOSIM train의 "text"로 하여 Llama_2를 지도학습. epoch=1, batch_size=32
  - step3 : llama 파인튜닝을 위해서 파일 수정함 -->  "KooJM/ATCOSIM_llm_train_text" (<s>[INST] 질문(Whisper-small 받아쓰기) [/INST] 정답(ATCOSIM train의 "text") </s>)
  - step3 : 파인튜닝 된 Llama_2를 허깅페이스에 올림 "KooJM/llama-2-7b_finetuned_using_whisper-small"
  - step4 : 파인튜닝 된 Llama_2를 허깅페이스에서 다운받아서 ATCOSIM test 받아쓰기한걸 llama-2에 prompt
  - step5 : llama-2가 뱉는 text와 ATCOSIM test 정답지를 비교하여 WER 계산!
![image](https://github.com/AIDL-final-project-2024-1st/Transformer_from_scratch/assets/170100329/eec6f55b-6f07-4230-83d6-fb8bcb80d095)
  - step6 : llama-2가 뱉는 text와 ATCO2 정답지를 비교하여 WER 계산해야함...

-------------------------
파인튜닝 안한 whisper small 의 atcosim 결과    

![image](https://github.com/AIDL-final-project-2024-1st/Transformer_from_scratch/assets/93754352/4704e93e-11fa-42e4-b153-e876e3cc0d87)

whisper small을 atcosim_converted 으로 파인튜닝한 결과. 1000에폭

![image](https://github.com/AIDL-final-project-2024-1st/Transformer_from_scratch/assets/93754352/216cab57-9c05-4e86-8e39-30ac2daafaf9)






--------------------------

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



