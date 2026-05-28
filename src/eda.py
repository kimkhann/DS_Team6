"""
UNSW-NB15 Exploratory Data Analysis (EDA)
==========================================
사용법:
    python eda_unsw_nb15.py

출력물:
    콘솔 출력 — shape, dtypes, head, 결측값, 중복행, 레이블 분포, 고상관 쌍
    eda_01_label_distribution.png
    eda_02_attack_cat_distribution.png
    eda_03_missing_values.png
    eda_04_duplicate_analysis.png
    eda_05_label_corr_ranking.png
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 설정
# ============================================================
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'UNSW-NB15_4.csv')

COLUMNS = [
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes',
    'sttl','dttl','sloss','dloss','service','sload','dload','spkts','dpkts',
    'swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len',
    'sjit','djit','stime','ltime','sintpkt','dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login',
    'ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm',
    'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','label'
]

CATEGORICAL = ['srcip','dstip','proto','state','service','attack_cat','label']
NUMERICAL   = [c for c in COLUMNS if c not in CATEGORICAL]

PALETTE = {0: '#61afef', 1: '#e06c75'}
sns.set_theme(style='whitegrid', font_scale=1.0)


# ============================================================
# 0. 데이터 로드
# ============================================================
def load_data(path):
    print("=" * 60)
    print("[LOAD] 데이터 로드 중...")
    df = pd.read_csv(path, header=None, names=COLUMNS, low_memory=False)

    # dsport: hex 문자열 → 정수 변환
    df['dsport'] = df['dsport'].apply(
        lambda x: int(str(x), 16) if str(x).startswith('0x') else pd.to_numeric(x, errors='coerce')
    )

    # ct_ftp_cmd: 공백 문자열 → 정수 변환
    df['ct_ftp_cmd'] = pd.to_numeric(
        df['ct_ftp_cmd'].astype(str).str.strip(), errors='coerce'
    ).fillna(0).astype(int)

    print(f"  로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
    print("=" * 60)
    return df


# ============================================================
# 1. 데이터셋 Shape
# ============================================================
def print_shape(df):
    print("\n[1] 데이터셋 Shape")
    print(f"  행(Row): {df.shape[0]:,}")
    print(f"  열(Col): {df.shape[1]}")


# ============================================================
# 2. 컬럼별 데이터 타입
# ============================================================
def print_dtypes(df):
    print("\n[2] 컬럼별 데이터 타입")
    dtype_df = pd.DataFrame({
        'dtype': df.dtypes,
        'category': ['범주형' if c in CATEGORICAL else '수치형' for c in df.columns]
    })
    print(dtype_df.to_string())
    print(f"\n  수치형: {(dtype_df['category']=='수치형').sum()}개")
    print(f"  범주형: {(dtype_df['category']=='범주형').sum()}개")


# ============================================================
# 3. df.head(5)
# ============================================================
from tabulate import tabulate

def print_head(df):
    print("\n[3] df.head(5)")
    head_T = df.head(5).T.reset_index()
    head_T.columns = ['Feature'] + [f'Row{i}' for i in range(5)]
    print(tabulate(head_T, headers='keys', tablefmt='simple', showindex=False, maxcolwidths=20))

# ============================================================
# 4. 결측값 — 열 종류, 개수, 비율 + 시각화
# ============================================================
def analyze_missing(df):
    print("\n[4] 결측값 현황")

    # ct_ftp_cmd는 이미 load 시점에 변환했으므로 isnull로 잡힘
    missing_count = df.isnull().sum()
    missing_pct   = missing_count / len(df) * 100
    missing_df    = pd.DataFrame({
        '결측 개수': missing_count,
        '결측 비율(%)': missing_pct.round(2)
    })
    missing_df = missing_df[missing_df['결측 개수'] > 0].sort_values('결측 비율(%)', ascending=False)

    if missing_df.empty:
        print("  결측값 없음")
    else:
        print(missing_df.to_string())

    # 시각화
    if not missing_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#e06c75' if v > 50 else '#e5c07b' if v > 10 else '#98c379'
                  for v in missing_df['결측 비율(%)'].values]
        bars = ax.barh(missing_df.index[::-1], missing_df['결측 비율(%)'].values[::-1],
                       color=colors[::-1], edgecolor='white')
        for bar, val in zip(bars, missing_df['결측 비율(%)'].values[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10)
        ax.set_title('Missing Value Rate by Feature', fontsize=13, fontweight='bold')
        ax.set_xlabel('Missing Rate (%)')
        ax.set_xlim(0, 115)
        plt.tight_layout()
        plt.savefig('eda_03_missing_values.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  저장: eda_03_missing_values.png")


# ============================================================
# 5. 중복행 — 개수, 비율 + 시각화
# ============================================================
def analyze_duplicates(df):
    print("\n[5] 중복행 현황")

    total_dup  = df.duplicated().sum()
    normal_dup = df[df['label'] == 0].duplicated().sum()
    attack_dup = df[df['label'] == 1].duplicated().sum()

    dup_df = pd.DataFrame({
        '전체 행': [len(df), (df['label']==0).sum(), (df['label']==1).sum()],
        '중복 행': [total_dup, normal_dup, attack_dup],
        '중복 비율(%)': [
            round(total_dup / len(df) * 100, 1),
            round(normal_dup / (df['label']==0).sum() * 100, 1),
            round(attack_dup / (df['label']==1).sum() * 100, 1),
        ]
    }, index=['전체', 'Normal (label=0)', 'Attack (label=1)'])
    print(dup_df.to_string())

    # 시각화
    categories = ['Total', 'Normal (label=0)', 'Attack (label=1)']
    dup_rates  = dup_df['중복 비율(%)'].values
    totals     = dup_df['전체 행'].values
    dups       = dup_df['중복 행'].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(categories))
    w = 0.35
    axes[0].bar(x - w/2, totals, w, label='Total',      color='#abb2bf', edgecolor='white')
    axes[0].bar(x + w/2, dups,   w, label='Duplicates', color='#e06c75', edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=9)
    axes[0].set_title('Duplicate Count', fontsize=12)
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    for i, (t, d) in enumerate(zip(totals, dups)):
        axes[0].text(i + w/2, d + 1000, f'{d:,}', ha='center', fontsize=8)

    colors = ['#abb2bf', '#61afef', '#e06c75']
    bars = axes[1].bar(categories, dup_rates, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, dup_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', fontsize=10)
    axes[1].set_title('Duplicate Rate (%)', fontsize=12)
    axes[1].set_ylabel('%')
    axes[1].set_ylim(0, max(dup_rates) * 1.25)

    plt.suptitle('Duplicate Row Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_04_duplicate_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: eda_04_duplicate_analysis.png")


# ============================================================
# 6. 레이블 분포 + 공격 유형 분포 시각화
# ============================================================
def analyze_label_distribution(df):
    print("\n[6] 레이블 분포")

    label_counts = df['label'].value_counts().sort_index()
    print(f"  Normal (0): {label_counts[0]:,} ({label_counts[0]/len(df)*100:.1f}%)")
    print(f"  Attack (1): {label_counts[1]:,} ({label_counts[1]/len(df)*100:.1f}%)")
    print(f"  불균형 비율: {label_counts[0]/label_counts[1]:.2f}:1")

    # 레이블 분포 시각화
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars = axes[0].bar(
        ['Normal (0)', 'Attack (1)'], label_counts.values,
        color=[PALETTE[i] for i in label_counts.index],
        edgecolor='white', width=0.5
    )
    for bar, val in zip(bars, label_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000,
                     f'{val:,}', ha='center', va='bottom', fontsize=10)
    axes[0].set_title('Label Distribution (Count)', fontsize=12)
    axes[0].set_ylabel('Count')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    axes[1].pie(
        label_counts.values,
        labels=['Normal (0)', 'Attack (1)'],
        colors=[PALETTE[i] for i in label_counts.index],
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    axes[1].set_title('Label Distribution (Ratio)', fontsize=12)
    plt.suptitle('Label Distribution — Original Data', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('eda_01_label_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: eda_01_label_distribution.png")

    # 공격 유형 분포 시각화
    attack_counts = df[df['label'] == 1]['attack_cat'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette('Set2', len(attack_counts))
    bars = ax.barh(attack_counts.index[::-1], attack_counts.values[::-1],
                   color=colors[::-1], edgecolor='white')
    for bar, val in zip(bars, attack_counts.values[::-1]):
        ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height()/2,
                f'{val:,} ({val/attack_counts.sum()*100:.1f}%)', va='center', fontsize=9)
    ax.set_title('Attack Category Distribution — Original Data', fontsize=13, fontweight='bold')
    ax.set_xlabel('Count')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_xlim(0, attack_counts.max() * 1.25)
    plt.tight_layout()
    plt.savefig('eda_02_attack_cat_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: eda_02_attack_cat_distribution.png")


# ============================================================
# 7. label과 상관관계 분포 + 시각화
# ============================================================
def analyze_label_correlation(df):
    print("\n[7] label과의 상관관계")

    num_df = df[NUMERICAL + ['label']].select_dtypes(include=[np.number])
    corr = num_df.corr()['label'].drop('label').sort_values()

    print(corr.to_string())
    print(f"\n  |r| > 0.3 피처: {(corr.abs() > 0.3).sum()}개")
    print(f"  |r| < 0.01 피처 (제거 대상): {(corr.abs() < 0.01).sum()}개")
    print(f"    → {corr[corr.abs() < 0.01].index.tolist()}")

    # 시각화
    colors = ['#e06c75' if v > 0 else '#61afef' for v in corr.values]
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.barh(corr.index, corr.values, color=colors, edgecolor='white')
    ax.axvline(0,     color='black', linewidth=0.8)
    ax.axvline(0.01,  color='gray',  linewidth=0.8, linestyle='--', alpha=0.6,
               label='|r|=0.01 (removal threshold)')
    ax.axvline(-0.01, color='gray',  linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_title('Correlation with Label (Pearson r)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pearson r')
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig('eda_05_label_corr_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: eda_05_label_corr_ranking.png")


# ============================================================
# 8. 피처 간 고상관 쌍
# ============================================================
def analyze_high_correlation(df):
    print("\n[8] 피처 간 고상관 쌍 (|r| > 0.95)")

    num_df = df[NUMERICAL].select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()

    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.95:
                pairs.append((cols[i], cols[j], round(r, 3)))

    if not pairs:
        print("  고상관 쌍 없음")
    else:
        pairs_df = pd.DataFrame(pairs, columns=['Feature A', 'Feature B', 'r'])
        pairs_df = pairs_df.sort_values('r', key=abs, ascending=False)
        print(f"  총 {len(pairs_df)}쌍")
        print(pairs_df.to_string(index=False))


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    df = load_data(FILE_PATH)

    print_shape(df)
    print_dtypes(df)
    print_head(df)
    analyze_missing(df)
    analyze_duplicates(df)
    analyze_label_distribution(df)
    analyze_label_correlation(df)
    analyze_high_correlation(df)

    print("\n" + "=" * 60)
    print("EDA 완료. 생성된 파일:")
    for f in [
        "eda_01_label_distribution.png",
        "eda_02_attack_cat_distribution.png",
        "eda_03_missing_values.png",
        "eda_04_duplicate_analysis.png",
        "eda_05_label_corr_ranking.png",
    ]:
        print(f"  {f}")
    print("=" * 60)
