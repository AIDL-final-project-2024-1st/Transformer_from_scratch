#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 1 21:42:04 ~ 
@author: yuminho
"""

###########################################################################
# ATCO2 test 파일을 모두 열어서 XML 파일 내 속성을 읽고 해당 내용을 seaborn 시각화 표현
###########################################################################
import os
import xml.etree.ElementTree as ET
import pandas as pd
import seaborn as sns


def parse_xml_to_dict(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    data = []

    for child in root:
        row_data = {elem.tag: elem.text for elem in child}
        for data_element in root.findall('segment/tags'):
            row_data2 = {elem.tag: elem.text for elem in data_element}
        row_data.update(row_data2)
        data.append(row_data)
    

    return data


def collect_data_from_folder(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    all_data = []
    
    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        file_data = parse_xml_to_dict(file_path)
        all_data.extend(file_data)
    
    combined_df = pd.DataFrame(all_data)
    
    return combined_df

# 예시 사용법
folder_path = '/Users/yuminho/Desktop/yonsei/third_semester/세미나/ATCO2-ASRdataset-v1_beta/DATA'
combined_df = collect_data_from_folder(folder_path)

# 데이터프레임 출력
print(combined_df.head())

# 각 항목별 분석 예시
# 예를 들어, 특정 컬럼의 통계 분석
if not combined_df.empty:
    print(combined_df.describe())
    
import matplotlib.pyplot as plt

#각 시간의 차이를 구하기 위함
combined_df['end'] = combined_df['end'].astype(float)
combined_df['start'] = combined_df['start'].astype(float)
combined_df['duration'] = combined_df['end'] - combined_df['start']
#duration 관련 정보
print( round(combined_df['duration'].mean(),2) )
print( round(combined_df['duration'].std(),2 ) )
print( round(combined_df['duration'].min(),2 ) )
print( round(combined_df['duration'].max(),2 ) )\
#duration 그래프 그리기
sns.histplot(x=combined_df['duration'])


#스피커 라벨에 대하여 그래프 그리기
sns.histplot(x=combined_df['speaker_label'], bins=100)
plt.xticks([]) # 눈금 레이블 제거
plt.show() # 그래프 표시


#각 스피커별 전사데이터 그래프 그리기
sns.histplot(x=combined_df['speaker'])
#각 스피커별 전사데이터 구하기
value_counts = combined_df['speaker'].value_counts()
print(value_counts)


# 유니크한 단어의 수를 구해보자
# 모든 텍스트를 결합하여 단어 목록 생성
all_text = ' '.join(combined_df['text'])
all_words = all_text.split()
# 유니크한 단어 집합 생성
unique_words = set(all_words)
# 유니크한 단어 수 계산
unique_word_count = len(unique_words)
print("\nTotal unique word count:")
print(unique_word_count)

# 전체 단어의 총합을 구해보자
# 텍스트 열의 각 행에 있는 단어 수를 계산
combined_df['word_count'] = combined_df['text'].apply(lambda x: len(x.split()))
# 모든 단어 수의 총합 계산
total_word_count = combined_df['word_count'].sum()
print("\nTotal word count:")
print(total_word_count)

#기타 data field값인 tag에 대해서 확인해보자. 
sns.countplot(x = combined_df['correct'])
sns.countplot(x = combined_df['correct_transcript'])
sns.countplot(x = combined_df['correct_tagging'])
sns.countplot(x = combined_df['non_english'])


###########################################################################
# 파일을 모두 열어서 해당 파일들의 크기의 평균을 구하는 공식
###########################################################################
import os
import xml.etree.ElementTree as ET
import pandas as pd
import seaborn as sns

data = []
def file_size(file_path):
    file_size = os.path.getsize(file_path) 
    return file_size

def collect_data_from_folder(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    all_data = []
    
    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        file_data = file_size(file_path)
        data.append(file_data)
    
    return data

# 예시 사용법
folder_path = '/Users/yuminho/Desktop/yonsei/third_semester/세미나/ATCO2-ASRdataset-v1_beta/DATA'
combined_data = collect_data_from_folder(folder_path)
mean = (sum(combined_data) / len(combined_data)) / 1024
print('File Size mean is', mean,  'Kbytes')

###########################################################################
# ATCOSIM 데이터 분석
###########################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '/Users/yuminho/Desktop/yonsei/third_semester/세미나/ATCOSIM/TXTdata/fulldata.csv'

# CSV 파일을 판다스 데이터프레임에 로드
df = pd.read_csv(file_path)

#duration 관련 정보
print( round(df['length_sec'].mean(),2) )
print( round(df['length_sec'].std(),2 ) )
print( round(df['length_sec'].min(),2 ) )
print( round(df['length_sec'].max(),2 ) )

#duration 그래프 그리기
sns.histplot(x=df['length_sec'])

#각 스피커별 전사데이터 그래프 그리기
sns.histplot(x=df['speaker_id'])
#각 스피커별 전사데이터 구하기
df_value_counts = df['speaker_id'].value_counts()
print(df_value_counts)

# 유니크한 단어의 수를 구해보자
# 모든 텍스트를 결합하여 단어 목록 생성
df_all_text = ' '.join(df['transcription'])
df_all_words = df_all_text.split()
# 유니크한 단어 집합 생성
df_unique_words = set(df_all_words)
# 유니크한 단어 수 계산
df_unique_word_count = len(df_unique_words)
print("\nTotal unique word count:")
print(df_unique_word_count)


# 전체 단어의 총합을 구해보자
# 텍스트 열의 각 행에 있는 단어 수를 계산
df['word_count'] = df['transcription'].apply(lambda x: len(x.split()))
# 모든 단어 수의 총합 계산
df_total_word_count = df['word_count'].sum()
print("\nTotal word count:")
print(df_total_word_count)


