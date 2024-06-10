
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
folder_path = '/Users/yuminho/Desktop/yonsei/third_semester/세미나/ATCO2-ASRdataset-v1_beta/DATA'
combined_df = collect_data_from_folder(folder_path)

# 데이터프레임 출력
print(combined_df.head())

# 각 항목별 분석 예시
# 예를 들어, 특정 컬럼의 통계 분석
if not combined_df.empty:
    print(combined_df.describe())
    
import matplotlib.pyplot as plt

sns.histplot(x=combined_df['speaker_label'], bins=100)

# 눈금 레이블 제거
plt.xticks([])

# 그래프 표시
plt.show()


sns.countplot(x = combined_df['correct'])
sns.countplot(x = combined_df['correct_transcript'])
sns.countplot(x = combined_df['correct_tagging'])
sns.countplot(x = combined_df['non_english'])