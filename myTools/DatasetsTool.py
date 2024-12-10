import pandas as pd
from collections import defaultdict

# 读取CSV文件
file_path = 'path/to/your/file.csv'
df = pd.read_csv(file_path, header=None)

# 构建problem-skill集合
problem_skill_map = defaultdict(set)

# 解析每4行的数据
num_students = len(df) // 4
for i in range(num_students):
    skill_row = df.iloc[4 * i + 1].dropna().astype(int)
    problem_row = df.iloc[4 * i + 2].dropna().astype(int)

    # 构建problem-skill集合
    for skill, problem in zip(skill_row, problem_row):
        problem_skill_map[problem].add(skill)

# 输出结果
for problem in sorted(problem_skill_map):
    skills = sorted(problem_skill_map[problem])
    print(f"{problem}: {' '.join(map(str, skills))}")