复现时间序列Transformer模型的步骤：

1、下载完整的Chinese diabetes datasets数据集
2、使用covert_xlsx.py将所有数据集格式统一转化为xlsx文件
3、使用pre-process.py分别对T1DM和T2DM进行数据预处理，记得修改summary_path和cgm_folder_path路径

或者可以直接使用我给的T1DM和T2DM压缩包，里面是预处理好的数据

4、使用model_transformer.py进行模型的训练、测试、结果可视化，可随意调参