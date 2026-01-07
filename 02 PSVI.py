#!/usr/bin/env python
# coding: utf-8

# In[1]:


#计算每个county的PSVI SCORE RATING
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# === 1. 读取数据 ===
file_path = r"D:\000database\eaglei_outages_2024.csv"
df = pd.read_csv(file_path)

# 确保有必要的列
df = df[['fips_code','county','state','customers_out','total_customers','run_start_time']]

# 转换时间
df['run_start_time'] = pd.to_datetime(df['run_start_time'])

# === 2. 计算 outage rate ===
df['outage_rate'] = df['customers_out'] / df['total_customers']

# 0.1% 作为 outage 判定阈值
df['is_outage'] = df['outage_rate'] >= 0.001

# === 3. 识别 outage events（简化版：连续 outage 判定为一个事件）===
events = []
for fips, group in df.groupby("fips_code"):
    group = group.sort_values("run_start_time")
    in_event = False
    event_start = None
    event_customers = []
    
    for idx, row in group.iterrows():
        if row['is_outage']:
            if not in_event:
                in_event = True
                event_start = row['run_start_time']
                event_customers = []
            event_customers.append(row['outage_rate'])
        else:
            if in_event:
                # 事件结束
                event_end = row['run_start_time']
                duration = (event_end - event_start).total_seconds() / 3600.0  # 小时
                events.append({
                    "fips_code": fips,
                    "start": event_start,
                    "end": event_end,
                    "duration_h": duration,
                    "max_outage_rate": max(event_customers) if event_customers else 0,
                    "mean_outage_rate": np.mean(event_customers) if event_customers else 0
                })
                in_event = False
    # 处理最后未关闭的事件
    if in_event:
        event_end = group.iloc[-1]['run_start_time']
        duration = (event_end - event_start).total_seconds() / 3600.0
        events.append({
            "fips_code": fips,
            "start": event_start,
            "end": event_end,
            "duration_h": duration,
            "max_outage_rate": max(event_customers) if event_customers else 0,
            "mean_outage_rate": np.mean(event_customers) if event_customers else 0
        })

df_events = pd.DataFrame(events)

# === 4. 聚合到 county-level 特征 ===
features = df_events.groupby("fips_code").agg({
    "duration_h": ["mean","max","sum"],
    "max_outage_rate": ["mean","max"],
    "mean_outage_rate": ["mean"],
})
features.columns = ["_".join(col).strip() for col in features.columns.values]
features.reset_index(inplace=True)

# === 5. 归一化 + 计算 PSVI ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features.drop(columns=['fips_code']))
features["PSVI_score"] = X_scaled.mean(axis=1)

# === 6. 分级（分位数法，5 类）===
features["PSVI_rating"] = pd.qcut(features["PSVI_score"], 5,
                                  labels=["Minor","Moderate","Major","Severe","Extreme"])

# === 7. 保存结果 ===
output_path = r"D:\000database\county_level_psvi.csv"
features.to_csv(output_path, index=False)

print(f"✅ 已生成 {output_path}")


# In[10]:


#生成热力图 score
import pandas as pd
import plotly.express as px

# 读取 PSVI 数据
features = pd.read_csv(r"D:\000database\county_level_psvi.csv")
features['fips_code_str'] = features['fips_code'].astype(str).str.zfill(5)

# === 连续热力图（PSVI_score） ===
fig = px.choropleth(
    features,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="fips_code_str",
    color="PSVI_score",
    color_continuous_scale="Reds",
    scope="usa",
    labels={'PSVI_score':'PSVI Score'}
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title="County-level PSVI Score Heatmap")

# 保存为 html，自动在浏览器打开
fig.write_html("D:/000database/psvi_score_map.html", auto_open=True)


# In[11]:


#生成热力图 rating
fig2 = px.choropleth(
    features,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="fips_code_str",
    color="PSVI_rating",
    color_discrete_map={
        "Minor":"#d4f0f0",
        "Moderate":"#a2d5d5",
        "Major":"#f9d57e",
        "Severe":"#f08c4f",
        "Extreme":"#d73027"
    },
    scope="usa",
    labels={'PSVI_rating':'PSVI Rating'}
)
fig2.update_geos(fitbounds="locations", visible=False)
fig2.update_layout(title="County-level PSVI Rating Heatmap")

fig2.write_html("D:/000database/psvi_rating_map.html", auto_open=True)


# In[12]:


#将PSVI对应到每个county的学校数量
import pandas as pd

# === 1. 读取两个文件 ===
psvi_path = r"D:\000database\county_level_psvi.csv"
county_path = r"D:\000database\high_schools_per_county.csv"

df_psvi = pd.read_csv(psvi_path, dtype={"fips_code": str})
df_county = pd.read_csv(county_path, dtype={"FIPS": str})

# === 2. 合并（按县 FIPS 匹配） ===
df_merged = pd.merge(
    df_county,
    df_psvi[["fips_code", "PSVI_score", "PSVI_rating"]],
    left_on="FIPS",
    right_on="fips_code",
    how="left"
)

# === 3. 保存结果 ===
output_path = r"D:\000database\high_schools_with_psvi.csv"
df_merged.to_csv(output_path, index=False)

print("✅ 合并完成！结果保存到:", output_path)
print(df_merged.head())


# In[14]:


#生成了热力图
import pandas as pd
import plotly.express as px

# === 1. 读取合并好的数据 ===
merged_path = r"D:\000database\high_schools_with_psvi.csv"
df = pd.read_csv(merged_path, dtype={"FIPS": str})

# 确保 FIPS 是 5 位字符串
df["FIPS"] = df["FIPS"].str.zfill(5)

# === 2. 绘制地图（按 PSVI_score 着色） ===
fig = px.choropleth(
    df,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="FIPS",
    color="PSVI_score",
    color_continuous_scale="Reds",
    range_color=(0, 1),
    scope="usa",
    labels={"PSVI_score": "PSVI Score"},
    hover_data=["County", "StateName", "SchoolCount", "PSVI_rating"]
)

# === 3. 添加气泡（表示 SchoolCount） ===
fig_bubble = px.scatter_geo(
    df,
    lat=df["LAT"] if "LAT" in df.columns else None,   # 如果有经纬度就用
    lon=df["LON"] if "LON" in df.columns else None,
    scope="usa",
    size="SchoolCount",
    hover_name="County",
    hover_data=["StateName", "SchoolCount", "PSVI_rating"],
    opacity=0.6
)

for trace in fig_bubble.data:
    fig.add_trace(trace)

# === 4. 保存为 HTML 文件 ===
output_path = r"D:\000database\psvi_map.html"
fig.write_html(output_path)

print(f"✅ 已生成地图文件：{output_path} （双击用浏览器打开查看）")


# In[15]:


#只筛选出major severe extreme三种
import pandas as pd

# 1. 读取文件
file_path = r"D:\000database\high_schools_with_psvi.csv"
df = pd.read_csv(file_path, dtype={"FIPS": str})

# 2. 删除 PSVI_rating 为空的行
df_clean = df.dropna(subset=["PSVI_rating"])

# 3. 只保留 Major / Severe / Extreme
df_filtered = df_clean[df_clean["PSVI_rating"].isin(["Major","Severe","Extreme"])]

# 4. 保存新文件
output_path = r"D:\000database\high_schools_with_psvi_major_severe_extreme.csv"
df_filtered.to_csv(output_path, index=False)

print(f"✅ 已保留 Major/Severe/Extreme 并保存到: {output_path}")
print(df_filtered.head())


# In[18]:


#添加气候区
import pandas as pd

# 1. 读取学校数据
schools_path = r"D:\000database\high_schools_with_psvi_major_severe_extreme.csv"
df_schools = pd.read_csv(schools_path, dtype={"FIPS": str})
df_schools["FIPS"] = df_schools["FIPS"].str.zfill(5)  # 确保 5 位字符串

# 2. 读取气候区数据
climate_path = r"D:\000database\county_climate_zone.csv"
df_climate = pd.read_csv(climate_path, dtype={"State FIPS": str, "County FIPS": str})

# 生成完整县级 FIPS（州码 + 县码）
df_climate["State FIPS"] = df_climate["State FIPS"].str.zfill(2)
df_climate["County FIPS"] = df_climate["County FIPS"].str.zfill(3)
df_climate["FIPS_full"] = df_climate["State FIPS"] + df_climate["County FIPS"]

# 3. 合并
df_merged = pd.merge(
    df_schools,
    df_climate[['FIPS_full', 'IECC Climate Zone']],
    left_on='FIPS',
    right_on='FIPS_full',
    how='left'
)

# 4. 保存
output_path = r"D:\000database\high_schools_with_psvi_major_severe_extreme_climate.csv"
df_merged.to_csv(output_path, index=False)

print(f"✅ 已将气候区信息添加并保存到：{output_path}")
print(df_merged[['FIPS','IECC Climate Zone']].head())


# In[31]:


# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import geopandas as gpd
import matplotlib.pyplot as plt

# === 1. 读取高中数据 ===
file_path = r"D:\000database\high_schools_with_psvi_major_severe_extreme_climate.csv"
df = pd.read_csv(file_path, dtype={"FIPS": str})

# 删除缺失值
df = df.dropna(subset=["PSVI_rating","IECC Climate Zone"])

# === 2. One-Hot 编码（PSVI_rating + 气候区）===
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(df[["PSVI_rating","IECC Climate Zone"]])

# === 3. KMeans 聚类 ===
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(X_encoded)

# === 4. 保存聚类结果 ===
output_path = r"D:\000database\high_schools_with_psvi_clusters.csv"
df.to_csv(output_path, index=False)
print(f"✅ 聚类完成并保存到 {output_path}")


# In[53]:


#
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# 读取 GeoJSON
geojson_path = r"D:\000database\counties.json"
gdf_counties = gpd.read_file(geojson_path, encoding='latin1')

# 生成 fips_code
gdf_counties["STATE"] = gdf_counties["STATE"].astype(str).str.zfill(2)
gdf_counties["COUNTY"] = gdf_counties["COUNTY"].astype(str).str.zfill(3)
gdf_counties["fips_code"] = gdf_counties["STATE"] + gdf_counties["COUNTY"]

# 读取县级聚类结果
county_cluster_path = r"D:\000database\high_schools_with_psvi_clusters.csv"
df_clusters = pd.read_csv(county_cluster_path, dtype={"FIPS": str})

# 聚合到县级（每县最多簇作为该县簇）
county_cluster = df_clusters.groupby("FIPS")["cluster"].agg(lambda x: x.value_counts().idxmax()).reset_index()
county_cluster.rename(columns={"FIPS": "fips_code"}, inplace=True)

# 合并
gdf_map = gdf_counties.merge(county_cluster, on="fips_code", how="left")

# 把 cluster 列转换为字符串类型（离散类别）
gdf_map["cluster_str"] = gdf_map["cluster"].astype(str)

# 自定义颜色映射
cluster_colors = {
    "0": "#499BC0",  # 蓝
    "1": "#8FDEE3",  # 橙
    "2": "#FDD786",  # 绿
    "3": "#F78779",  # 红
}

# 绘图
fig, ax = plt.subplots(figsize=(24, 16))
gdf_map.boundary.plot(ax=ax, color="gray", linewidth=0.5)
gdf_map.plot(column="cluster", cmap="tab10", legend=True, ax=ax, missing_kwds={"color": "lightgray"})

# 设置图形标题和坐标轴
plt.title("County-level Clustering of High Schools (PSVI + Climate Zone)", fontsize=18)
plt.xlabel("Longitude", fontsize=14)
plt.ylabel("Latitude", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 限制 X 轴范围
ax.set_xlim([-130, -60])
ax.set_ylim([22, 50])
plt.show()


# In[55]:


#计算每个聚类的学校总数
import pandas as pd

# 读取聚类结果
file_path = r"D:\000database\high_schools_with_psvi_clusters.csv"
df = pd.read_csv(file_path, dtype={"FIPS": str})

# 按 cluster 汇总 SchoolCount
cluster_schoolcount = df.groupby("cluster")["SchoolCount"].sum().sort_index()

# 输出结果
print("每个 cluster 的学校总数：")
print(cluster_schoolcount)

# 如果想保存到 CSV
cluster_schoolcount.to_csv(r"D:\000database\high_school_cluster_total_counts.csv", header=["TotalSchoolCount"])


# In[56]:


import pandas as pd

# 读取聚类后的高中数据
file_path = r"D:\000database\high_schools_with_psvi_clusters.csv"
df = pd.read_csv(file_path, dtype={"FIPS": str})

# 汇总每个 cluster 的学校总数
school_counts = df.groupby("cluster")["SchoolCount"].sum().reset_index()
school_counts.rename(columns={"SchoolCount": "Total_Schools"}, inplace=True)

# 统计每个 cluster 中 PSVI_rating 和 Climate Zone 分布
cluster_summary = []

for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    
    psvi_dist = sub.groupby("PSVI_rating")["SchoolCount"].sum().to_dict()
    climate_dist = sub.groupby("IECC Climate Zone")["SchoolCount"].sum().to_dict()
    
    cluster_summary.append({
        "cluster": c,
        "Total_Schools": school_counts.loc[school_counts["cluster"]==c, "Total_Schools"].values[0],
        "PSVI_distribution": psvi_dist,
        "Climate_distribution": climate_dist
    })

# 转成 DataFrame 便于查看
summary_df = pd.DataFrame(cluster_summary)
print(summary_df)


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# 读取高中聚类文件
file_path = r"D:\000database\high_schools_with_psvi_clusters.csv"
df = pd.read_csv(file_path, dtype={"FIPS": str})

# 计算每个 cluster 的学校总数（SchoolCount 列求和）
cluster_summary = df.groupby("cluster")["SchoolCount"].sum().reset_index()
cluster_summary.rename(columns={"SchoolCount": "Total_Schools"}, inplace=True)

# 计算每个 cluster 的 PSVI 分布
psvi_dist = []
for cluster, group in df.groupby("cluster"):
    psvi_counter = dict(group.groupby("PSVI_rating")["SchoolCount"].sum())
    psvi_dist.append(psvi_counter)
cluster_summary["PSVI_distribution"] = psvi_dist

# 计算每个 cluster 的气候区分布
climate_dist = []
for cluster, group in df.groupby("cluster"):
    climate_counter = dict(group.groupby("IECC Climate Zone")["SchoolCount"].sum())
    climate_dist.append(climate_counter)
cluster_summary["Climate_distribution"] = climate_dist

# --------------------- 绘图 ---------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 一行两列

# --------------------- 左图：PSVI 分布 ---------------------
ax = axes[0]
all_psvi = set()
for d in cluster_summary["PSVI_distribution"]:
    all_psvi.update(d.keys())
all_psvi = sorted(list(all_psvi))

bottoms = [0] * len(cluster_summary)
colors = ["#499BC0", "#8FDEE3", "#FDD786", "#F78779"]
for i, psvi in enumerate(all_psvi):
    heights = [d.get(psvi, 0) for d in cluster_summary["PSVI_distribution"]]
    ax.bar(cluster_summary["cluster"], heights, bottom=bottoms, label=psvi, color=colors[i % len(colors)])
    bottoms = [b + h for b, h in zip(bottoms, heights)]

ax.set_xlabel("Cluster", fontsize=16)
ax.set_ylabel("Number of Schools", fontsize=16)
ax.set_title("PSVI Distribution by Cluster", fontsize=18)
ax.legend(title="PSVI Rating", fontsize=16, title_fontsize=16)

# --------------------- 右图：气候区分布 ---------------------
ax = axes[1]
all_climate = set()
for d in cluster_summary["Climate_distribution"]:
    all_climate.update(d.keys())
all_climate = sorted(list(all_climate))

bottoms = [0] * len(cluster_summary)
colors = plt.cm.tab20.colors  # 足够多的颜色
for i, climate in enumerate(all_climate):
    heights = [d.get(climate, 0) for d in cluster_summary["Climate_distribution"]]
    ax.bar(cluster_summary["cluster"], heights, bottom=bottoms, label=f"Zone {climate}", color=colors[i % len(colors)])
    bottoms = [b + h for b, h in zip(bottoms, heights)]

ax.set_xlabel("Cluster", fontsize=16)
ax.set_ylabel("Number of Schools", fontsize=16)
ax.set_title("Climate Zone Distribution by Cluster", fontsize=18)
ax.legend(title="Climate Zone", fontsize=16, title_fontsize=16, loc='upper right')

plt.tight_layout()
plt.show()


# In[ ]:




