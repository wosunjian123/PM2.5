import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示与图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# ----------------------
# 1. 基础配置：文件路径与监测点定义
# ----------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
# 五城市数据文件路径
file_path_dic = {
    "Beijing": os.path.join(script_dir, "PM2.5data", "BeijingPM20100101_20151231.csv"),
    "Shanghai": os.path.join(script_dir, "PM2.5data", "ShanghaiPM20100101_20151231.csv"),
    "Chengdu": os.path.join(script_dir, "PM2.5data", "ChengduPM20100101_20151231.csv"),
    "Guangzhou": os.path.join(script_dir, "PM2.5data", "GuangzhouPM20100101_20151231.csv"),
    "Shenyang": os.path.join(script_dir, "PM2.5data", "ShenyangPM20100101_20151231.csv")
}
# 各城市中国环保部监测点（核心：用于后续观测点差异分析）
city_china_monitors = {
    "Beijing": ["PM_Dongsi", "PM_Dongsihuan", "PM_Nongzhanguan"],
    "Shanghai": ["PM_Jingan", "PM_Xuhui"],
    "Chengdu": ["PM_Caotangsi", "PM_Shahepu"],
    "Guangzhou": ["PM_City Station", "PM_5th Middle School"],
    "Shenyang": ["PM_Taiyuanjie", "PM_Xiaoheyan"]
}
us_col = "PM_US Post"  # 美国大使馆统一监测点
result_dir = "result"  # 结果保存目录
os.makedirs(result_dir, exist_ok=True)  # 自动创建目录（不存在时）


# ----------------------
# 2. 数据加载与预处理
# ----------------------
def load_city_data(file_path_dic):
    """批量加载五城市数据，添加日期与年月列"""
    city_dfs = {}
    for city, path in file_path_dic.items():
        try:
            # 加载原始数据
            df = pd.read_csv(path)
            # 合并年月日为日期格式（用于按天/按月分组）
            df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
            # 提取“年月”（用于月度差异分析，格式：2010-01）
            df["year_month"] = df["date"].dt.to_period("M")
            # 过滤无效日期数据
            df = df.dropna(subset=["date", "year_month"])
            city_dfs[city] = df
            print(f"✅ {city}数据加载完成：时间范围{df['date'].min().date()}~{df['date'].max().date()}，有效行数{len(df)}")
        except FileNotFoundError:
            print(f"❌ 未找到{city}数据文件，路径：{path}")
        except Exception as e:
            print(f"⚠️ {city}数据加载异常：{str(e)}")
    return city_dfs

# 执行数据加载
city_dfs = load_city_data(file_path_dic)
if not city_dfs:
    print("❌ 无有效城市数据，程序终止")
    exit()


# ----------------------
# 3. 核心分析1：每个城市每个观测点的月度差异
# ----------------------
def calc_station_monthly_avg(city_df, city_name, china_monitors):
    """
    计算单个城市每个观测点的月度平均PM2.5
    返回：月度平均值DataFrame（行：年月，列：观测点）
    """
    # 按“年月+观测点”分组，计算每月平均（排除NaN）
    monthly_data = []
    for station in china_monitors:
        # 过滤该观测点的有效数据
        station_df = city_df.dropna(subset=[station])
        if len(station_df) == 0:
            print(f"⚠️ {city_name}-{station}无有效数据，跳过")
            continue
        # 按年月分组计算平均
        station_monthly = station_df.groupby("year_month")[station].mean().reset_index()
        station_monthly.rename(columns={station: f"{station}"}, inplace=True)
        monthly_data.append(station_monthly)
    
    # 合并所有观测点的月度数据（按年月对齐）
    if monthly_data:
        monthly_avg = monthly_data[0]
        for df in monthly_data[1:]:
            monthly_avg = pd.merge(monthly_avg, df, on="year_month", how="outer")
        # 转换年月为字符串（便于绘图）
        monthly_avg["year_month_str"] = monthly_avg["year_month"].astype(str)
        return monthly_avg
    return None

def plot_station_monthly_diff(city_name, monthly_avg, china_monitors):
    """绘制单个城市各观测点的月度PM2.5对比折线图"""
    if monthly_avg is None:
        return
    # 筛选有效观测点（排除无数据的点）
    valid_stations = [col for col in china_monitors if col in monthly_avg.columns]
    if len(valid_stations) < 2:
        print(f"⚠️ {city_name}有效观测点不足2个，无法绘制月度差异图")
        return
    
    # 创建图表
    plt.figure(figsize=(14, 7))
    # 定义颜色（区分不同观测点）
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]
    
    # 绘制每个观测点的月度趋势
    for i, station in enumerate(valid_stations):
        # 过滤该观测点的NaN数据
        plot_data = monthly_avg.dropna(subset=[station])
        if len(plot_data) == 0:
            continue
        # 绘制折线
        plt.plot(
            plot_data["year_month_str"], plot_data[station],
            label=station.replace("PM_", ""),  # 简化标签（去掉PM_前缀）
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=4,
            linewidth=2,
            alpha=0.8
        )
    
    # 图表美化
    plt.xlabel("年月", fontsize=12)
    plt.ylabel("PM2.5浓度（μg/m³）", fontsize=12)
    plt.title(f"{city_name}各观测点PM2.5月度平均值对比（2010-2015）", fontsize=14, pad=20)
    plt.legend(loc="upper right", fontsize=10)
    # 优化x轴标签（每6个月显示一个，避免重叠）
    plt.xticks(
        range(0, len(monthly_avg["year_month_str"]), 6),
        monthly_avg["year_month_str"][::6],
        rotation=45
    )
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    # 保存图表
    save_path = os.path.join(result_dir, f"{city_name}_各观测点月度PM2.5对比.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 {city_name}观测点月度差异图已保存：{save_path}")

# 批量执行“观测点月度差异分析”
city_station_monthly = {}  # 存储各城市观测点月度数据
for city, df in city_dfs.items():
    # 计算月度平均
    monthly_avg = calc_station_monthly_avg(
        city_df=df,
        city_name=city,
        china_monitors=city_china_monitors[city]
    )
    city_station_monthly[city] = monthly_avg
    # 绘制月度差异图
    plot_station_monthly_diff(
        city_name=city,
        monthly_avg=monthly_avg,
        china_monitors=city_china_monitors[city]
    )
    # 导出月度数据到CSV（实验报告可引用）
    if monthly_avg is not None:
        monthly_avg.to_csv(
            os.path.join(result_dir, f"{city}_各观测点月度PM2.5数据.csv"),
            index=False
        )


# ----------------------
# 4. 核心分析2：每个城市每天的平均PM2.5（中美双口径）与折线图
# ----------------------
def calc_city_daily_avg(city_df, china_monitors, us_col):
    """
    计算单个城市的每日平均PM2.5
    - 中国口径：所有本土观测点的日均平均值（消除单观测点误差）
    - 美国口径：美国大使馆单观测点日均值
    """
    # 1. 中国环保部口径：多观测点日均平均
    china_daily = city_df.groupby("date")[china_monitors].mean()  # 每个观测点的日均值
    china_daily["China_Avg"] = china_daily.mean(axis=1)  # 所有观测点的日均平均（最终中国口径）
    
    # 2. 美国大使馆口径：单观测点日均值
    us_daily = city_df.groupby("date")[us_col].mean().rename("US_Avg")
    
    # 3. 合并双口径数据，保留至少一个有效值的日期
    daily_avg = pd.concat([china_daily["China_Avg"], us_daily], axis=1).dropna(how="all")
    # 添加年份列（用于后续按年分组绘图）
    daily_avg["year"] = daily_avg.index.year
    return daily_avg

def plot_city_daily_avg(city_name, daily_avg):
    """绘制单个城市的每日PM2.5折线图（中美双口径+按年区分）"""
    if len(daily_avg) == 0:
        print(f"⚠️ {city_name}无有效每日数据，无法绘制折线图")
        return
    
    # 创建图表
    plt.figure(figsize=(16, 8))
    # 按年份分组绘制（避免单条线过于密集）
    years = sorted(daily_avg["year"].unique())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    # 绘制中国口径每日平均（按年着色）
    for i, year in enumerate(years):
        year_data = daily_avg[daily_avg["year"] == year]
        plt.plot(
            year_data.index, year_data["China_Avg"],
            label=f"中国环保部-{year}",
            color=colors[i % len(colors)],
            linewidth=1.5,
            alpha=0.9
        )
    
    # 绘制美国口径每日平均（统一用黑色虚线，突出对比）
    us_valid = daily_avg.dropna(subset=["US_Avg"])
    if len(us_valid) > 0:
        plt.plot(
            us_valid.index, us_valid["US_Avg"],
            label="美国驻华大使馆",
            color="#000000",
            linestyle="--",
            linewidth=2,
            alpha=0.8
        )
    
    # 添加超标线（中国PM2.5日均标准：75μg/m³）
    plt.axhline(y=75, color="red", linestyle="-.", linewidth=1.5, label="超标线（75μg/m³）")
    
    # 图表美化
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("PM2.5浓度（μg/m³）", fontsize=12)
    plt.title(f"{city_name}每日平均PM2.5浓度趋势（中美双口径对比）", fontsize=14, pad=20)
    plt.legend(loc="upper left", fontsize=10, ncol=2)
    # 优化x轴（按年显示刻度）
    plt.xticks(
        pd.date_range(start=daily_avg.index.min(), end=daily_avg.index.max(), freq="YS"),
        [d.strftime("%Y") for d in pd.date_range(start=daily_avg.index.min(), end=daily_avg.index.max(), freq="YS")],
        rotation=0
    )
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    # 保存图表
    save_path = os.path.join(result_dir, f"{city_name}_每日PM2.5折线图.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 {city_name}每日PM2.5折线图已保存：{save_path}")

# 批量执行“每日平均计算与折线图绘制”
city_daily_avg = {}  # 存储各城市每日平均数据
for city, df in city_dfs.items():
    # 计算每日平均
    daily_avg = calc_city_daily_avg(
        city_df=df,
        china_monitors=city_china_monitors[city],
        us_col=us_col
    )
    city_daily_avg[city] = daily_avg
    print(f"📈 {city}每日平均计算完成：有效天数{len(daily_avg)}，中国口径均值{daily_avg['China_Avg'].mean():.2f}μg/m³")
    
    # 绘制每日折线图
    plot_city_daily_avg(city_name=city, daily_avg=daily_avg)
    
    # 导出每日数据到CSV（实验报告可直接引用）
    daily_avg_export = daily_avg.reset_index()
    daily_avg_export["date"] = daily_avg_export["date"].dt.date  # 简化日期格式（仅保留年月日）
    daily_avg_export.to_csv(
        os.path.join(result_dir, f"{city}_每日PM2.5平均值.csv"),
        index=False,
        columns=["date", "China_Avg", "US_Avg", "year"]  # 仅保留核心列
    )


# ----------------------
# 5. 辅助：中美污染等级一致性统计（保留原功能，确保完整性）
# ----------------------
def get_pollution_level(pm_value):
    """中国《环境空气质量标准》PM2.5污染等级划分"""
    if pd.isna(pm_value):
        return None
    elif pm_value <= 35:
        return "优"
    elif pm_value <= 75:
        return "良"
    elif pm_value <= 115:
        return "轻度污染"
    elif pm_value <= 150:
        return "中度污染"
    else:
        return "重度污染"

# 统计各城市中美等级一致性
consistency_summary = []
for city, daily_avg in city_daily_avg.items():
    # 计算污染等级
    daily_avg["China_Level"] = daily_avg["China_Avg"].apply(get_pollution_level)
    daily_avg["US_Level"] = daily_avg["US_Avg"].apply(get_pollution_level)
    # 筛选有效对比天数
    valid_days = daily_avg.dropna(subset=["China_Level", "US_Level"])
    if len(valid_days) == 0:
        continue
    # 计算一致性指标
    consistent_days = (valid_days["China_Level"] == valid_days["US_Level"]).sum()
    consistency_rate = (consistent_days / len(valid_days)) * 100
    consistency_summary.append({
        "城市": city,
        "有效对比天数": len(valid_days),
        "等级一致天数": consistent_days,
        "等级不一致天数": len(valid_days) - consistent_days,
        "一致率(%)": round(consistency_rate, 2),
        "中国优天数": (valid_days["China_Level"] == "优").sum(),
        "美国优天数": (valid_days["US_Level"] == "优").sum()
    })

# 导出一致性汇总表（实验报告核心表格）
if consistency_summary:
    consistency_df = pd.DataFrame(consistency_summary)
    consistency_df.to_csv(
        os.path.join(result_dir, "中美污染等级一致性汇总.csv"),
        index=False
    )
    print(f"📋 中美等级一致性汇总表已保存：{os.path.join(result_dir, '中美污染等级一致性汇总.csv')}")


# ----------------------
# 6. 五城污染状态分析：等级分布统计与可视化
# ----------------------
levels_order = ["优", "良", "轻度污染", "中度污染", "重度污染"]

# 逐城统计中/美两种口径的等级分布（天数与占比），并导出CSV与图表
china_dist_rows = []
us_dist_rows = []
for city, daily_avg in city_daily_avg.items():
    # 确保存在等级列（若前面一致性统计步骤未执行到此处，也在此补齐）
    if "China_Level" not in daily_avg.columns:
        daily_avg["China_Level"] = daily_avg["China_Avg"].apply(get_pollution_level)
    if "US_Level" not in daily_avg.columns:
        daily_avg["US_Level"] = daily_avg["US_Avg"].apply(get_pollution_level)

    # 中国口径分布
    china_valid = daily_avg.dropna(subset=["China_Level"])  # 仅统计有等级的天
    china_counts = china_valid["China_Level"].value_counts().reindex(levels_order, fill_value=0)
    china_total = china_counts.sum()
    china_perc = (china_counts / china_total * 100).round(2) if china_total > 0 else pd.Series([0]*5, index=levels_order)

    # 美国口径分布
    us_valid = daily_avg.dropna(subset=["US_Level"])  # 仅统计有等级的天
    us_counts = us_valid["US_Level"].value_counts().reindex(levels_order, fill_value=0)
    us_total = us_counts.sum()
    us_perc = (us_counts / us_total * 100).round(2) if us_total > 0 else pd.Series([0]*5, index=levels_order)

    # 导出当前城市等级分布CSV
    city_dist_df = pd.DataFrame({
        "等级": levels_order,
        "中国_天数": china_counts.values,
        "中国_占比(%)": china_perc.values,
        "美国_天数": us_counts.values,
        "美国_占比(%)": us_perc.values
    })
    city_dist_path = os.path.join(result_dir, f"{city}_污染等级分布_中美对比.csv")
    city_dist_df.to_csv(city_dist_path, index=False)
    print(f"📄 {city}污染等级分布表已保存：{city_dist_path}")

    # 汇总到五城分布汇总（分别汇总中国与美国口径，便于跨城比较）
    china_row = {"城市": city}
    us_row = {"城市": city}
    for lvl in levels_order:
        china_row[f"{lvl}_占比(%)"] = float(china_perc[lvl]) if china_total > 0 else 0.0
        us_row[f"{lvl}_占比(%)"] = float(us_perc[lvl]) if us_total > 0 else 0.0
    china_dist_rows.append(china_row)
    us_dist_rows.append(us_row)

    # 绘制当前城市中美等级分布对比柱状图
    x = np.arange(len(levels_order))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, china_perc.values, width=width, label="中国环保部", color="#4E79A7")
    plt.bar(x + width/2, us_perc.values, width=width, label="美国大使馆", color="#F28E2B")
    plt.xticks(x, levels_order)
    plt.ylabel("占比（%）")
    plt.title(f"{city}污染等级分布（中美口径对比）")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig_path = os.path.join(result_dir, f"{city}_污染等级分布_中美对比.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"📊 {city}污染等级分布图已保存：{fig_path}")

# 五城汇总表（按城市行，列为各等级占比）
china_summary_df = pd.DataFrame(china_dist_rows)
us_summary_df = pd.DataFrame(us_dist_rows)

china_summary_path = os.path.join(result_dir, "五城市污染等级分布_中国口径.csv")
us_summary_path = os.path.join(result_dir, "五城市污染等级分布_美国口径.csv")
china_summary_df.to_csv(china_summary_path, index=False)
us_summary_df.to_csv(us_summary_path, index=False)
print(f"🗂️ 五城市等级分布汇总表已保存：{china_summary_path} / {us_summary_path}")

# 五城堆叠柱状图（中国口径）
def plot_city_stack(summary_df, title, save_name):
    cities = summary_df["城市"].tolist()
    x = np.arange(len(cities))
    plt.figure(figsize=(12, 7))
    bottom = np.zeros(len(cities))
    colors = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"]
    for i, lvl in enumerate(levels_order):
        vals = summary_df[f"{lvl}_占比(%)"].values
        plt.bar(x, vals, bottom=bottom, label=lvl, color=colors[i % len(colors)])
        bottom += vals
    plt.xticks(x, cities)
    plt.ylabel("占比（%）")
    plt.title(title)
    plt.ylim(0, 100)
    plt.legend(title="等级")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_path = os.path.join(result_dir, save_name)
    plt.savefig(out_path)
    plt.close()
    print(f"📈 五城市堆叠图已保存：{out_path}")

plot_city_stack(china_summary_df, "五城市污染等级分布（中国环保部口径）", "五城市污染等级分布_中国口径_堆叠.png")
plot_city_stack(us_summary_df, "五城市污染等级分布（美国大使馆口径）", "五城市污染等级分布_美国口径_堆叠.png")


print("\n🎉 所有分析完成！结果文件已保存至：", os.path.abspath(result_dir))