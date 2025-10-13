import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置dir
file_path_dic = {
    # 北京文件直接用文件名（因为和脚本在同一目录）
    "Beijing": os.path.join(script_dir, "PM2.5data","BeijingPM20100101_20151231.csv"),
    # 其他城市仍从PM2.5data文件夹读取（如果未移动）
    "Shanghai": os.path.join(script_dir, "PM2.5data", "ShanghaiPM20100101_20151231.csv"),
    "Chengdu": os.path.join(script_dir, "PM2.5data", "ChengduPM20100101_20151231.csv"),
    "Guangzhou": os.path.join(script_dir, "PM2.5data", "GuangzhouPM20100101_20151231.csv"),
    "Shenyang": os.path.join(script_dir, "PM2.5data", "ShenyangPM20100101_20151231.csv")
}

#导入五个城市的PM2.5数据
try:
    df_bj = pd.read_csv(file_path_dic["Beijing"])
    df_sh = pd.read_csv(file_path_dic["Shanghai"])
    df_cd = pd.read_csv(file_path_dic["Chengdu"])
    df_gz = pd.read_csv(file_path_dic["Guangzhou"])
    df_sy = pd.read_csv(file_path_dic["Shenyang"])
except FileNotFoundError:
    print(f"错误：找不到文件")

print(df_gz.head())
print(type(df_bj["season"][0]))
print(type(df_gz["season"][0]))

print(f"bj:",df_bj.columns)
print(f"sh:",df_sh.columns)
print(f"cd:",df_cd.columns)
print(f"gz:",df_gz.columns)
print(f"sy:",df_sy.columns)


#需要对NONE进行处理
bj_pm_mean = df_bj.loc[df_bj["PM_Dongsi"].notnull(),"PM_Dongsi"].mean()
sh_pm_mean = df_sh.loc[df_sh["PM_Jingan"].notnull(),"PM_Jingan"].mean()
cd_pm_mean = df_cd.loc[df_cd["PM_Caotangsi"].notnull(),"PM_Caotangsi"].mean()
gz_pm_mean = df_gz.loc[df_gz["PM_City Station"].notnull(),"PM_City Station"].mean()
sy_pm_mean = df_sy.loc[df_sy["PM_Taiyuanjie"].notnull(),"PM_Taiyuanjie"].mean()


print(df_bj["PM_Dongsi"].groupby(df_bj["year"]).mean())




print(df_bj["PM_Dongsi"].groupby(df_bj["year"]).mean())
df_bj_year_mean = df_bj.loc[df_bj["PM_Dongsi"].notnull(),"PM_Dongsi"].groupby(df_bj["year"]).mean()
df_sh_year_mean = df_sh.loc[df_sh["PM_Dongsi"].notnull(),"PM_Dongsi"].groupby(df_sh["year"]).mean()
df_cd_year_mean = df_cd.loc[df_cd["PM_Dongsi"].notnull(),"PM_Dongsi"].groupby(df_cd["year"]).mean()
df_gz_year_mean = df_gz.loc[df_gz["PM_Dongsi"].notnull(),"PM_Dongsi"].groupby(df_gz["year"]).mean()
df_sy_year_mean = df_sy.loc[df_sy["PM_Dongsi"].notnull(),"PM_Dongsi"].groupby(df_sy["year"]).mean()
print(df_bj.loc[df_bj["PM_Dongsi"].notnull(),"PM_Dongsi"].groupby(df_bj["year"]).mean())