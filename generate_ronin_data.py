import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置随机种子
np.random.seed(42)

# ============================
# 1. 创建文件夹结构
# ============================
folders = [
    "data/raw",
    "data/processed",
    "models",
    "src",
    "dashboard",
    "docs",
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# src 下创建空的 __init__.py
init_path = os.path.join("src", "__init__.py")
with open(init_path, "w", encoding="utf-8") as f:
    pass

# ============================
# 2. 生成 respondents.csv
# ============================
n = 1000

respondent_ids = [f"R{i:03d}" for i in range(1, n + 1)]

# industry: 权重 [25%, 20%, 25%, 15%, 15%]
industries = np.random.choice(
    ['Finance', 'Healthcare', 'Technology', 'Retail', 'Manufacturing'],
    size=n,
    p=[0.25, 0.20, 0.25, 0.15, 0.15]
)

# job_level: 严格比例 C-Suite 15%, Director 25%, Manager 35%, Specialist 25%
job_levels = (
    ['C-Suite'] * 150 +
    ['Director'] * 250 +
    ['Manager'] * 350 +
    ['Specialist'] * 250
)
np.random.shuffle(job_levels)

# region: 各 25%
regions = (
    ['APAC'] * 250 +
    ['EMEA'] * 250 +
    ['North America'] * 250 +
    ['LATAM'] * 250
)
np.random.shuffle(regions)

# company_size: 权重 [30%, 40%, 30%]
company_sizes = np.random.choice(
    ['Enterprise', 'Mid-market', 'SMB'],
    size=n,
    p=[0.30, 0.40, 0.30]
)

# past_participation_count: 70%是0，20%是1-2，10%大于2
past_counts = np.zeros(n, dtype=int)
r = np.random.rand(n)
mask_0 = r < 0.70
mask_1_2 = (r >= 0.70) & (r < 0.90)
mask_gt2 = r >= 0.90

past_counts[mask_0] = 0
past_counts[mask_1_2] = np.random.choice([1, 2], size=mask_1_2.sum())
past_counts[mask_gt2] = np.random.choice([3, 4, 5], size=mask_gt2.sum())

# preferred_contact: 权重 [50%, 30%, 20%]
preferred_contacts = np.random.choice(
    ['Email', 'LinkedIn', 'Phone'],
    size=n,
    p=[0.50, 0.30, 0.20]
)

# research_topic_match_score: 正态分布 N(60, 15)，限制 0-100，取整
scores = np.random.normal(loc=60, scale=15, size=n)
scores = np.clip(scores, 0, 100).round().astype(int)

# is_hard_to_reach: C-Suite 且 Healthcare = 1
df_resp = pd.DataFrame({
    'respondent_id': respondent_ids,
    'industry': industries,
    'job_level': job_levels,
    'region': regions,
    'company_size': company_sizes,
    'past_participation_count': past_counts,
    'preferred_contact': preferred_contacts,
    'research_topic_match_score': scores,
})

df_resp['is_hard_to_reach'] = (
    (df_resp['job_level'] == 'C-Suite') & (df_resp['industry'] == 'Healthcare')
).astype(int)

# 调整列顺序
df_resp = df_resp[[
    'respondent_id', 'industry', 'job_level', 'region', 'company_size',
    'past_participation_count', 'preferred_contact',
    'research_topic_match_score', 'is_hard_to_reach'
]]

# ============================
# 3. 生成 interactions.csv
# ============================
# 每个受访者 3-8 条记录，总共恰好 5000 条
total_interactions = 5000
min_count, max_count = 3, 8

# 先均匀采样，再微调至总和为 5000
counts = np.random.choice(range(min_count, max_count + 1), size=n)
diff = counts.sum() - total_interactions

while diff != 0:
    if diff > 0:
        # 需要减少记录数
        reducible = np.where(counts > min_count)[0]
        if len(reducible) == 0:
            break
        idx = np.random.choice(reducible)
        counts[idx] -= 1
        diff -= 1
    else:
        # 需要增加记录数
        increasable = np.where(counts < max_count)[0]
        if len(increasable) == 0:
            break
        idx = np.random.choice(increasable)
        counts[idx] += 1
        diff += 1

# 构建 respondent_id 列表
interaction_respondent_ids = np.repeat(df_resp['respondent_id'].values, counts)

# contact_date: 2024-01-01 到 2024-12-31 随机
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
days_range = (end_date - start_date).days
random_days = np.random.randint(0, days_range + 1, size=total_interactions)
contact_dates = [start_date + timedelta(days=int(d)) for d in random_days]

# channel: 70% preferred_contact, 30% 随机其他两个之一
channels = []
contact_map = df_resp.set_index('respondent_id')['preferred_contact'].to_dict()
other_options = {
    'Email': ['LinkedIn', 'Phone'],
    'LinkedIn': ['Email', 'Phone'],
    'Phone': ['Email', 'LinkedIn'],
}

for rid in interaction_respondent_ids:
    preferred = contact_map[rid]
    if np.random.rand() < 0.70:
        channels.append(preferred)
    else:
        channels.append(np.random.choice(other_options[preferred]))

# action_type 权重基础值
# sent=100, opened=40, clicked=15, scheduled=8, declined=12, no_response=20
# replied 根据规则动态调整
action_types = []
replied_flags = []

hard_to_reach_map = df_resp.set_index('respondent_id')['is_hard_to_reach'].to_dict()
past_count_map = df_resp.set_index('respondent_id')['past_participation_count'].to_dict()

base_weights = {
    'sent': 100,
    'opened': 40,
    'clicked': 15,
    'scheduled': 8,
    'declined': 12,
    'no_response': 20,
}

for rid in interaction_respondent_ids:
    htr = hard_to_reach_map[rid]
    pc = past_count_map[rid]
    
    if htr == 1:
        replied_w = 12
    elif pc > 0:
        replied_w = 65
    else:
        replied_w = 25
    
    weights = list(base_weights.values()) + [replied_w]
    labels = list(base_weights.keys()) + ['replied']
    
    action = np.random.choice(labels, p=np.array(weights) / np.sum(weights))
    action_types.append(action)
    replied_flags.append(1 if action == 'replied' else 0)

# time_to_response_hours: replied 则 1-72，否则 NA
time_to_response = []
for is_replied in replied_flags:
    if is_replied:
        time_to_response.append(np.random.randint(1, 73))
    else:
        time_to_response.append(pd.NA)

df_inter = pd.DataFrame({
    'respondent_id': interaction_respondent_ids,
    'contact_date': [d.strftime('%Y-%m-%d') for d in contact_dates],
    'channel': channels,
    'action_type': action_types,
    'time_to_response_hours': time_to_response,
})

# ============================
# 4. 保存文件
# ============================
respondents_path = os.path.join("data", "raw", "respondents.csv")
interactions_path = os.path.join("data", "raw", "interactions.csv")

df_resp.to_csv(respondents_path, index=False, encoding='utf-8')
df_inter.to_csv(interactions_path, index=False, encoding='utf-8')

# requirements.txt
requirements_content = """pandas==2.1.4
numpy==1.24.3
lightgbm==4.1.0
scikit-learn==1.3.2
plotly==5.18.0
flask==3.0.0
sentence-transformers==2.2.2
langchain==0.1.0
faiss-cpu==1.7.4
openai==1.6.1
"""

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements_content)

# ============================
# 5. 打印统计信息
# ============================
print("=" * 50)
print("RONIN Digital Research - 模拟数据生成统计")
print("=" * 50)
print(f"总受访者数: {len(df_resp)}")
print(f"总交互记录数: {len(df_inter)}")
print()

print("【各 industry 占比】")
industry_pct = df_resp['industry'].value_counts(normalize=True).sort_index() * 100
for ind, pct in industry_pct.items():
    print(f"  {ind}: {pct:.1f}%")
print()

print("【各 job_level 占比】")
job_pct = df_resp['job_level'].value_counts(normalize=True).sort_index() * 100
for job, pct in job_pct.items():
    marker = "  <-- 重点" if job == 'C-Suite' else ""
    print(f"  {job}: {pct:.1f}%{marker}")
print()

print("【实际响应率统计 - 按 is_hard_to_reach 分组】")
# 计算每个受访者的响应率
df_inter['replied'] = (df_inter['action_type'] == 'replied').astype(int)
reply_rate = df_inter.groupby('respondent_id')['replied'].mean().reset_index()
reply_rate = reply_rate.merge(df_resp[['respondent_id', 'is_hard_to_reach']], on='respondent_id')
for htr_val, group in reply_rate.groupby('is_hard_to_reach'):
    label = "Hard-to-reach (医生/高管)" if htr_val == 1 else "普通受访者"
    print(f"  {label}: 平均响应率 = {group['replied'].mean()*100:.1f}%")
print()

print("【action_type 分布】")
action_pct = df_inter['action_type'].value_counts(normalize=True) * 100
for act, pct in action_pct.sort_index().items():
    print(f"  {act}: {pct:.1f}%")
print()

print("数据生成完成，位于 ./data/raw/")
print("=" * 50)
