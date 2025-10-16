import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_dic = {
    "Beijing": os.path.join(script_dir, "PM2.5data", "BeijingPM20100101_20151231.csv"),
    "Shanghai": os.path.join(script_dir, "PM2.5data", "ShanghaiPM20100101_20151231.csv"),
    "Chengdu": os.path.join(script_dir, "PM2.5data", "ChengduPM20100101_20151231.csv"),
    "Guangzhou": os.path.join(script_dir, "PM2.5data", "GuangzhouPM20100101_20151231.csv"),
    "Shenyang": os.path.join(script_dir, "PM2.5data", "ShenyangPM20100101_20151231.csv")
}

city_china_monitors = {
    "Beijing": ["PM_Dongsi", "PM_Dongsihuan", "PM_Nongzhanguan"],
    "Shanghai": ["PM_Jingan", "PM_Xuhui"],
    "Chengdu": ["PM_Caotangsi", "PM_Shahepu"],
    "Guangzhou": ["PM_City Station", "PM_5th Middle School"],
    "Shenyang": ["PM_Taiyuanjie", "PM_Xiaoheyan"]
}
us_col = "PM_US Post"  


# 2. 加载所有城市数据（批量处理，避免重复代码）
city_dfs = {}  # 存储所有城市原始数据
for city, path in file_path_dic.items():
    try:
        df = pd.read_csv(path)
        # 合并year/month/day为日期列（关键：用于按“天”分组）
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        city_dfs[city] = df
        print(f"{city}数据加载成功，时间范围：{df['date'].min()} ~ {df['date'].max()}，总行数：{len(df)}")
    except FileNotFoundError:
        print(f"错误：未找到{city}文件，路径：{path}")

# 示例：查看北京数据结构，确认日期列与监测点列
print("\n北京数据前3行（关键列）：")
print(city_dfs["Beijing"][["date", "year", "month", "day", "PM_Dongsi", "PM_US Post"]].head(3))

# 定义函数：计算单个城市的每日平均PM2.5（中国+美国）
def calc_daily_pm(city_df, china_monitor_cols, us_col):
    # 步骤1：按日期分组，计算中国环保部口径每日平均
    # 先计算每天每个本土监测点的平均（小时→天），再计算多个监测点的日均
    china_daily = city_df.groupby("date")[china_monitor_cols].mean()  # 每天各监测点均值
    china_daily["China_Avg"] = china_daily.mean(axis=1)  # 每天的本土监测点平均（最终中国口径）
    
    # 步骤2：按日期分组，计算美国大使馆口径每日平均
    us_daily = city_df.groupby("date")[us_col].mean().rename("US_Avg")  # 每天美国监测点均值
    
    # 步骤3：合并中国和美国的每日数据，删除全为NaN的行
    daily_df = pd.concat([china_daily["China_Avg"], us_daily], axis=1).dropna(how="all")
    return daily_df

# 批量计算所有城市的每日平均PM2.5
city_daily_pm = {}  # 存储各城市每日PM2.5（China_Avg + US_Avg）
for city, df in city_dfs.items():
    china_cols = city_china_monitors[city]
    daily_df = calc_daily_pm(df, china_cols, us_col)
    city_daily_pm[city] = daily_df
    print(f"\n{city}每日PM2.5统计完成：有效天数={len(daily_df)}，中国口径均值={daily_df['China_Avg'].mean():.2f}，美国口径均值={daily_df['US_Avg'].mean():.2f}")

# 示例：查看北京2010年1月前5天的每日数据
print("\n北京2010年1月每日PM2.5（中国vs美国）：")
beijing_daily = city_daily_pm["Beijing"]
print(beijing_daily[beijing_daily.index.month == 1].head())

# 1. 计算五城市核心统计指标
city_stats = []
for city, daily_df in city_daily_pm.items():
    # 有效数据量（排除NaN）
    china_valid = daily_df["China_Avg"].notna().sum()
    us_valid = daily_df["US_Avg"].notna().sum()
    # 中国口径统计
    china_mean = daily_df["China_Avg"].mean()
    china_over_75 = (daily_df["China_Avg"] > 75).sum()  # 超标天数（中国标准）
    china_over_rate = (china_over_75 / china_valid) * 100 if china_valid > 0 else 0
    # 美国口径统计（参考同一标准，便于对比）
    us_mean = daily_df["US_Avg"].mean()
    us_over_75 = (daily_df["US_Avg"] > 75).sum()
    us_over_rate = (us_over_75 / us_valid) * 100 if us_valid > 0 else 0
    
    city_stats.append({
        "城市": city,
        "中国口径有效天数": china_valid,
        "中国口径日均PM2.5(μg/m³)": round(china_mean, 2),
        "中国口径超标率(%)": round(china_over_rate, 2),
        "美国口径有效天数": us_valid,
        "美国口径日均PM2.5(μg/m³)": round(us_mean, 2),
        "美国口径超标率(%)": round(us_over_rate, 2)
    })

# 转为DataFrame，按“中国口径日均”排序（污染从重到轻）
city_stats_df = pd.DataFrame(city_stats).sort_values("中国口径日均PM2.5(μg/m³)", ascending=False)
print("\n五城市PM2.5统计对比表（中国vs美国）：")
print(city_stats_df)

# 2. 可视化：五城市中国口径日均PM2.5箱线图（展示分布与异常值）
plt.figure(figsize=(12, 6))
# 提取各城市中国口径数据（排除NaN）
china_data = [city_daily_pm[city]["China_Avg"].dropna() for city in city_stats_df["城市"]]
# 绘制箱线图
box = plt.boxplot(china_data, labels=city_stats_df["城市"], patch_artist=True)
# 美化：给箱体上色
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
# 添加标题与标签
plt.title("2010-2015年五城市PM2.5日均浓度分布（中国环保部口径）", fontsize=14)
plt.ylabel("PM2.5浓度（μg/m³）", fontsize=12)
plt.axhline(y=75, color="red", linestyle="--", label="超标线（75μg/m³）")  # 添加超标线
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.savefig("五城市PM2.5箱线图.png", dpi=300, bbox_inches="tight")
plt.close()


# 1. 年度趋势：计算各城市每年的平均PM2.5（中国口径）
city_yearly = {}
for city, daily_df in city_daily_pm.items():
    # 按“年”分组计算平均
    yearly = daily_df["China_Avg"].groupby(daily_df.index.year).mean()
    city_yearly[city] = yearly

# 可视化：五城市年度PM2.5趋势线
plt.figure(figsize=(12, 6))
for city, yearly_data in city_yearly.items():
    plt.plot(yearly_data.index, yearly_data.values, marker="o", label=city, linewidth=2)
# 添加标题与标签
plt.title("2010-2015年五城市PM2.5年度平均趋势（中国环保部口径）", fontsize=14)
plt.xlabel("年份", fontsize=12)
plt.ylabel("PM2.5浓度（μg/m³）", fontsize=12)
plt.axhline(y=75, color="red", linestyle="--", label="超标线（75μg/m³）")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("五城市PM2.5年度趋势.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. 季节趋势：计算各城市每季的平均PM2.5（中国口径，季节映射：1-3春，4-6夏，7-9秋，10-12冬）
def get_season(month):
    if month in [12, 1, 2]:
        return "冬季"
    elif month in [3, 4, 5]:
        return "春季"
    elif month in [6, 7, 8]:
        return "夏季"
    else:
        return "秋季"

# 批量计算各城市季节平均
city_seasonal = {}
for city, daily_df in city_daily_pm.items():
    # 添加季节列

    daily_df["season"] = pd.Series(daily_df.index.month).apply(get_season)
    # 按“季节”分组计算平均（固定季节顺序）
    seasonal = daily_df["China_Avg"].groupby(daily_df["season"]).mean()
    seasonal = seasonal.reindex(["春季", "夏季", "秋季", "冬季"])  # 按春夏秋冬排序
    city_seasonal[city] = seasonal

# 可视化：五城市季节PM2.5柱状图
plt.figure(figsize=(14, 8))
width = 0.15  # 柱子宽度
x = np.arange(4)  # 季节位置（春、夏、秋、冬）
cities = list(city_seasonal.keys())

# 绘制每个城市的四季柱状图
for i, city in enumerate(cities):
    seasonal_data = city_seasonal[city]
    plt.bar(x + i*width, seasonal_data.values, width=width, label=city)

# 添加标题与标签
plt.title("2010-2015年五城市PM2.5季节平均对比（中国环保部口径）", fontsize=14)
plt.xlabel("季节", fontsize=12)
plt.ylabel("PM2.5浓度（μg/m³）", fontsize=12)
plt.xticks(x + width*2, ["春季", "夏季", "秋季", "冬季"])  # 调整x轴标签位置
plt.axhline(y=75, color="red", linestyle="--", label="超标线（75μg/m³）")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.savefig("五城市PM2.5季节对比.png", dpi=300, bbox_inches="tight")
plt.close()


# 1. 批量计算各城市中美监测的相关性与误差
us_china_compare = []
for city, daily_df in city_daily_pm.items():
    # 仅保留两者均有数据的行（避免NaN影响）
    valid_df = daily_df.dropna(subset=["China_Avg", "US_Avg"])
    if len(valid_df) < 30:  # 有效数据不足30天，统计意义弱，跳过
        us_china_compare.append({"城市": city, "相关系数": "数据不足", "平均绝对误差": "数据不足", "平均相对误差(%)": "数据不足"})
        continue
    
    # 计算相关系数（Pearson相关，越接近1趋势越一致）
    corr = valid_df["China_Avg"].corr(valid_df["US_Avg"])
    # 平均绝对误差（MAE：|中国值-美国值|的平均，反映绝对差异）
    mae = np.abs(valid_df["China_Avg"] - valid_df["US_Avg"]).mean()
    # 平均相对误差（MRE：|中国值-美国值|/美国值 的平均，反映相对差异，避免因数值大小影响）
    mre = (np.abs(valid_df["China_Avg"] - valid_df["US_Avg"]) / valid_df["US_Avg"]).mean() * 100
    
    us_china_compare.append({
        "城市": city,
        "有效对比天数": len(valid_df),
        "相关系数": round(corr, 3),
        "平均绝对误差(μg/m³)": round(mae, 2),
        "平均相对误差(%)": round(mre, 2)
    })

# 转为DataFrame展示
compare_df = pd.DataFrame(us_china_compare)
print("\n中美监测结果对比表：")
print(compare_df)

# 2. 可视化：北京中美监测散点图（示例，其他城市可同理绘制）
plt.figure(figsize=(10, 8))
city = "Beijing"
valid_df = city_daily_pm[city].dropna(subset=["China_Avg", "US_Avg"])
# 绘制散点图
plt.scatter(valid_df["US_Avg"], valid_df["China_Avg"], alpha=0.6, color="#45B7D1")
# 添加对角线（y=x，代表中美数值完全一致）
max_val = max(valid_df["US_Avg"].max(), valid_df["China_Avg"].max())
plt.plot([0, max_val], [0, max_val], "r--", label="中美完全一致线（y=x）")
# 添加相关系数标注
corr = valid_df["China_Avg"].corr(valid_df["US_Avg"])
plt.annotate(f"Pearson相关系数: {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
# 添加标题与标签
plt.title(f"{city}中美PM2.5日均监测结果对比（2010-2015）", fontsize=14)
plt.xlabel("美国大使馆监测值（μg/m³）", fontsize=12)
plt.ylabel("中国环保部监测值（μg/m³）", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{city}中美监测散点图.png", dpi=300, bbox_inches="tight")
plt.close()


# 1. 定义函数：根据PM2.5值判断污染等级
def get_pollution_level(pm_val):
    if pd.isna(pm_val):
        return None
    elif pm_val <= 35:
        return "优"
    elif pm_val <= 75:
        return "良"
    elif pm_val <= 115:
        return "轻度污染"
    elif pm_val <= 150:
        return "中度污染"
    else:
        return "重度污染"

# 2. 批量计算各城市中美等级一致性
level_consistency = []
for city, daily_df in city_daily_pm.items():
    # 计算中美等级
    daily_df["China_Level"] = daily_df["China_Avg"].apply(get_pollution_level)
    daily_df["US_Level"] = daily_df["US_Avg"].apply(get_pollution_level)
    # 仅保留两者均有等级的行
    valid_level = daily_df.dropna(subset=["China_Level", "US_Level"])
    if len(valid_level) < 30:
        level_consistency.append({"城市": city, "有效对比天数": len(valid_level), "等级一致率(%)": "数据不足"})
        continue
    
    # 计算等级一致率（中国等级 == 美国等级的比例）
    consistent = (valid_level["China_Level"] == valid_level["US_Level"]).sum()
    consistent_rate = (consistent / len(valid_level)) * 100
    
    # 统计各等级的一致情况（可选，用于更细致分析）
    level_crosstab = pd.crosstab(valid_level["China_Level"], valid_level["US_Level"], margins=True, margins_name="总计")
    
    level_consistency.append({
        "城市": city,
        "有效对比天数": len(valid_level),
        "等级一致率(%)": round(consistent_rate, 2),
        "等级交叉表": level_crosstab
    })

# 3. 展示等级一致率
consistent_df = pd.DataFrame([{"城市": c["城市"], "有效对比天数": c["有效对比天数"], "等级一致率(%)": c["等级一致率(%)"]} for c in level_consistency])
print("\n中美污染等级一致性对比：")
print(consistent_df)

# 4. 可视化：北京中美等级一致性热力图（示例）
city = "Beijing"
for item in level_consistency:
    if item["城市"] == city and item["等级一致率(%)"] != "数据不足":
        crosstab = item["等级交叉表"].drop("总计", axis=0).drop("总计", axis=1)
        break

plt.figure(figsize=(10, 8))
# 定义等级顺序（按污染程度从低到高）
level_order = ["优", "良", "轻度污染", "中度污染", "重度污染"]
# 重新排序交叉表（确保热力图顺序正确）
crosstab = crosstab.reindex(index=level_order, columns=level_order).fillna(0)
# 绘制热力图
im = plt.imshow(crosstab.values, cmap="YlOrRd")
# 添加数值标注
for i in range(len(crosstab.index)):
    for j in range(len(crosstab.columns)):
        plt.text(j, i, int(crosstab.iloc[i, j]), ha="center", va="center", color="black" if crosstab.iloc[i, j] < 50 else "white")
# 设置坐标轴
plt.xticks(np.arange(len(crosstab.columns)), crosstab.columns, rotation=45)
plt.yticks(np.arange(len(crosstab.index)), crosstab.index)
# 添加标题与色条
plt.title(f"{city}中美污染等级判定交叉表（2010-2015）", fontsize=14)
plt.xlabel("美国大使馆判定等级", fontsize=12)
plt.ylabel("中国环保部判定等级", fontsize=12)
cbar = plt.colorbar(im)
cbar.set_label("天数", fontsize=12)
plt.tight_layout()
plt.savefig(f"{city}中美等级交叉热力图.png", dpi=300, bbox_inches="tight")
plt.close()