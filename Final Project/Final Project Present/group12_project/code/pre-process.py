import pandas as pd
import numpy as np
import os

# 加载静态特征数据
summary_path = 'Shanghai_T2DM_Summary.xlsx'
summary_df = pd.read_excel(summary_path)


def process_cgm_file(file_path, summary_df):
    cgm_df = pd.read_excel(file_path)
    patient_id_period = os.path.splitext(os.path.basename(file_path))[0]

    # 根据病人id找到表格
    static_features = summary_df[summary_df['Patient Number'] == patient_id_period]
    cgm_df['Patient_ID_Period'] = patient_id_period

    # 处理饮食数据
    columns_to_drop1 = ['Dietary intake', '饮食']
    columns_to_drop1 = [col for col in columns_to_drop1 if col in cgm_df.columns]

    if len(columns_to_drop1) == 2:
        # 添加新的take_food列为0-1变量
        cgm_df['take_food'] = ((cgm_df['Dietary intake'].notna()) | (cgm_df['饮食'].notna())).astype(int)
    else:
        cgm_df['take_food'] = (cgm_df['Dietary intake'].notna()).astype(int)

     # 处理一些与预测CGM无关的数据，指定要删除的列名
    columns_to_drop2 = ['Dietary intake', '饮食', 'CBG (mg / dl)', 'CSII - basal insulin (Novolin R, IU / H)',
                       'Blood Ketone (mmol / L)', '胰岛素泵基础量 (Novolin R, IU / H)']

    columns_to_drop2 = [col for col in columns_to_drop2 if col in cgm_df.columns]
    cgm_df = cgm_df.drop(columns=columns_to_drop2)

    # 处理动态分类特征，将其转换成0-1整数编码
    columns_to_drop3 = ['Insulin dose - s.c.', 'Insulin dose - i.v.', 'Non-insulin hypoglycemic agents', 'CSII - bolus insulin (Novolin R, IU)']

    columns_to_drop3 = [col for col in columns_to_drop3 if col in cgm_df.columns]
    for column in columns_to_drop3:
        cgm_df[column] = cgm_df[column].notna().astype(int)

    # 合并动态和静态特征
    merged_df = pd.merge(cgm_df, static_features, left_on='Patient_ID_Period', right_on='Patient Number', how='left')
    merged_df = merged_df.drop(columns=['Patient_ID_Period'])  # 删除Patient_ID_Period列

    # 将性别列中的1和2转换为0和1（女性为0，男性为1）
    merged_df['Gender (Female=0, Male=1)'] = (merged_df['Gender (Female=1, Male=2)'] - 1).astype(int)
    merged_df = merged_df.drop(columns=['Gender (Female=1, Male=2)'])

    # 移动Patient Number列到第一位
    cols = ['Patient Number'] + [col for col in merged_df.columns if col != 'Patient Number']
    merged_df = merged_df[cols]

    # 处理静态分类特征，将其转换成0-1整数编码
    merged_df['Alcohol Drinking History (drinker/non-drinker)'] = (
                merged_df['Alcohol Drinking History (drinker/non-drinker)'] == 'drinker').astype(int)  #酒精导致高血糖风险
    merged_df['Type of Diabetes'] = (merged_df['Type of Diabetes'] == 'T1DM').astype(int)  #T1DM相较于T2DM，高血糖风险较高
    merged_df['Acute Diabetic Complications'] = (
                merged_df['Acute Diabetic Complications'] == 'diabetic ketoacidosis').astype(int)  #高血糖伴随糖尿病酮症酸中毒
    merged_df['Hypoglycemia (yes/no)'] = (merged_df['Hypoglycemia (yes/no)'] == 'no').astype(int)  #低血糖症

    # 去除所有空列
    merged_df = merged_df.dropna(axis=1, how='all')

    # 去除所有除Date列以外的非数值列
    date_column = merged_df.iloc[:, 1]
    merged_df = merged_df.select_dtypes(include=[np.number])

    # 添加时间相关特征
    merged_df.insert(0, 'Date', date_column)
    merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%Y/%m/%d %H:%M:%S')

    # 添加周期性特征和滞后特征
    merged_df['hour'] = merged_df['Date'].dt.hour  # 小时
    merged_df['CGM_lag1'] = merged_df['CGM (mg / dl)'].shift(1)  # 前1期的血糖
    merged_df['CGM_lag2'] = merged_df['CGM (mg / dl)'].shift(2)  # 前2期的血糖
    print(patient_id_period)
    merged_df = merged_df.dropna(subset=['CGM_lag1', 'CGM_lag2'])  # 移除因创建滞后特征而产生的NaN值

    '''
    # 对于T1DM
    merged_df = merged_df.drop(columns=['Fasting Plasma Glucose (mg/dl)',
    'Total Cholesterol (mmol/L)', 'Triglyceride (mmol/L)',
    'High-Density Lipoprotein Cholesterol (mmol/L)',
    'Low-Density Lipoprotein Cholesterol (mmol/L)'])
    '''

    # 保存处理后的数据
    output_filename = f"processed_{os.path.splitext(os.path.basename(file_path))[0]}.xlsx"
    output_filepath = os.path.join(os.path.dirname(file_path), output_filename)
    merged_df.to_excel(output_filepath, index=False)


cgm_folder_path = 'Shanghai_T2DM_xlsx'
for filename in os.listdir(cgm_folder_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(cgm_folder_path, filename)
        process_cgm_file(file_path, summary_df)
