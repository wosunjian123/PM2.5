import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

dir = os.path.dirname(os.path.abspath(__file__))
# 1. 先创建result目录（不存在则创建，避免保存时路径报错）
result_dir = os.path.join(dir, "result")
os.makedirs(result_dir, exist_ok=True)  # exist_ok=True：目录已存在时不报错

file_path_dic = {
    "Beijing": os.path.join(dir, "PM2.5data", "BeijingPM20100101_20151231.csv"),
    "Shanghai": os.path.join(dir, "PM2.5data", "ShanghaiPM20100101_20151231.csv"),
    "Chengdu": os.path.join(dir, "PM2.5data", "ChengduPM20100101_20151231.csv"),
    "Guangzhou": os.path.join(dir, "PM2.5data", "GuangzhouPM20100101_20151231.csv"),
    "Shenyang": os.path.join(dir, "PM2.5data", "ShenyangPM20100101_20151231.csv")
}

city_china_monitors = {
    "Beijing": ["PM_Dongsi", "PM_Dongsihuan", "PM_Nongzhanguan"],
    "Shanghai": ["PM_Jingan", "PM_Xuhui"],
    "Chengdu": ["PM_Caotangsi", "PM_Shahepu"],
    "Guangzhou": ["PM_City Station", "PM_5th Middle School"],
    "Shenyang": ["PM_Taiyuanjie", "PM_Xiaoheyan"]
}
us_monitor = "PM_US Post" 

#存入城市数据
city_dfs = {}  
for city, path in file_path_dic.items():
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df.dropna(subset=city_china_monitors[city])
        city_avg_cn = df[city_china_monitors[city]].mean(axis=1).round(2)
        df['city_avg_cn'] = city_avg_cn
        city_dfs[city] = df
        city_dfs[city].to_csv(f"test{city}.csv")
    except FileNotFoundError:
        print(f"错误：未找到{city}文件，路径：{path}")

# 1. 计算每个城市的日均PM2.5
city_daily_data = {}  
for city, df in city_dfs.items():
    daily_avg = df.groupby("date")["city_avg_cn"].mean().round(2)
    daily_df = pd.DataFrame({
        "date": daily_avg.index,
        "daily_avg_cn": daily_avg.values
    }).set_index("date")
    city_daily_data[city] = daily_df

# 2. 筛选2014-2015年数据
city_2014_2015_data = {}
for city, daily_df in city_daily_data.items():
    mask = (daily_df.index.year >= 2014) & (daily_df.index.year <= 2015)
    target_df = daily_df[mask].copy()
    city_2014_2015_data[city] = target_df

# 3. 定义污染级别并添加
pm_bins = [0, 35, 75, 115, 150, 250, float('inf')]
pm_labels = ["优", "良", "轻度污染", "中度污染", "重度污染", "严重污染"]
for city in city_2014_2015_data:
    city_2014_2015_data[city]["level"] = pd.cut(
        city_2014_2015_data[city]["daily_avg_cn"],
        bins=pm_bins,
        labels=pm_labels,
        include_lowest=True
    )

# 4. 按年份和级别统计天数
city_level_stats = {}
for city, df in city_2014_2015_data.items():
    df["year"] = df.index.year
    stats = df.groupby(["year", "level"]).size().unstack(fill_value=0)
    city_level_stats[city] = stats

# 5. 打印统计结果
print("\n===== 2014-2015年各城市污染级别天数统计 =====")
for city, stats in city_level_stats.items():
    print(f"\n{city}：")
    print(stats)

# 6. 可视化：分年度子图（条形图上方显示天数）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
bar_width = 0.1
x = np.arange(len(pm_labels))
offset = 1.5  # 天数标签与条形顶部的偏移量

# 2014年子图
for i, city in enumerate(city_level_stats.keys()):
    stats = city_level_stats[city]
    y2014 = stats.loc[2014] if 2014 in stats.index else [0]*len(pm_labels)
    bars = ax1.bar(x + i*bar_width, y2014, width=bar_width, label=city)
    # 添加天数标签
    for bar, day_count in zip(bars, y2014):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + offset,
            str(int(day_count)),
            ha='center', va='bottom', fontsize=9
        )

ax1.set_title("2014年五城市PM2.5污染级别天数对比（国内监测点）", fontsize=14)
ax1.set_ylabel("天数", fontsize=12)
ax1.set_xticks(x + bar_width*(len(city_level_stats)-1)/2)
ax1.set_xticklabels(pm_labels, fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# 2015年子图
for i, city in enumerate(city_level_stats.keys()):
    stats = city_level_stats[city]
    y2015 = stats.loc[2015] if 2015 in stats.index else [0]*len(pm_labels)
    bars = ax2.bar(x + i*bar_width, y2015, width=bar_width, label=city)
    # 添加天数标签
    for bar, day_count in zip(bars, y2015):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + offset,
            str(int(day_count)),
            ha='center', va='bottom', fontsize=9
        )

ax2.set_title("2015年五城市PM2.5污染级别天数对比（国内监测点）", fontsize=14)
ax2.set_ylabel("天数", fontsize=12)
ax2.set_xlabel("污染级别", fontsize=12)
ax2.set_xticks(x + bar_width*(len(city_level_stats)-1)/2)
ax2.set_xticklabels(pm_labels, fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(axis='y', alpha=0.3)

# 2. 保存图片到result目录（必须在plt.show()之前，避免画布释放）
save_path = os.path.join(result_dir, "2014&2015_five_cities.png")
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi=300：高清图片；bbox_inches='tight'：避免标签被截断
plt.show()

# 7. 计算并打印占比
city_level_ratio = {}
for city, stats in city_level_stats.items():
    ratio = stats.div(stats.sum(axis=1), axis=0) * 100
    city_level_ratio[city] = ratio.round(1)

print("\n===== 2014-2015年各城市污染级别占比（%） =====")
for city, ratio in city_level_ratio.items():
    print(f"\n{city}：")
    print(ratio)


#分析五城市每个观测区空气质量的月度差异

# 8. 计算每个城市各观测点的月度平均值
# 存储每个城市各观测点的月度均值（格式：{城市: {观测点: 月度均值Series}}）
city_station_monthly = {}

for city in city_dfs.keys():
    # 获取该城市的原始数据和观测点列名
    df = city_dfs[city].copy()
    stations = city_china_monitors[city]
    
    # 提取年份和月份，用于分组
    df['year_month'] = df['date'].dt.to_period('M')  # 转换为年月格式（如2014-01）
    
    # 按年月和观测点计算月度均值
    station_monthly = {}
    for station in stations:
        # 按年月分组计算该观测点的月度平均值（排除NaN）
        monthly_avg = df.groupby('year_month')[station].mean().round(2)
        # 转换索引为 datetime 格式，便于绘图
        monthly_avg.index = monthly_avg.index.to_timestamp()
        station_monthly[station] = monthly_avg
    
    city_station_monthly[city] = station_monthly

# 9. 可视化：每个城市各观测点的月度均值折线图（按城市分子图）
# 创建2行3列的子图（5个城市 + 1个空图）
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()  # 转换为一维数组，便于索引

# 定义折线颜色和标记，区分不同观测点
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
markers = ['o', 's', '^', 'D', 'p']

for i, city in enumerate(city_station_monthly.keys()):
    ax = axes[i]
    station_data = city_station_monthly[city]
    stations = list(station_data.keys())
    
    # 为每个观测点绘制折线
    for j, station in enumerate(stations):
        monthly_data = station_data[station]
        ax.plot(
            monthly_data.index, 
            monthly_data.values,
            label=station,
            color=colors[j % len(colors)],
            marker=markers[j % len(markers)],
            markersize=4,
            linewidth=1.5
        )
    
    # 设置子图标题和标签
    ax.set_title(f'{city}各观测点PM2.5月度均值', fontsize=12)
    ax.set_xlabel('日期', fontsize=10)
    ax.set_ylabel('PM2.5浓度（μg/m³）', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 旋转x轴标签，避免重叠
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

# 隐藏最后一个空图（2行3列共6个子图，只需要5个）
axes[-1].axis('off')

# 调整布局
plt.tight_layout()
# 保存图片到result目录
save_path = os.path.join(result_dir, "各城市观测点月度均值对比.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()