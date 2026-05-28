"""
UNSW-NB15 Data Preprocessing Pipeline
======================================
사용법:
    python preprocessing_unsw_nb15.py

출력물:
    X_train.csv, X_test.csv
    y_train.csv, y_test.csv
    y_train_cat.csv, y_test_cat.csv   ← 군집화 평가용 attack_cat
    preprocessing_report.txt          ← 각 단계별 처리 결과 로그
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 설정
# ============================================================
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'UNSW-NB15_4.csv')
RANDOM_STATE = 42
TEST_SIZE    = 0.2
US_RATIO     = 3       # 언더샘플링 정상:공격 비율
VAR_THRESH   = 0.01    # VarianceThreshold 기준
CORR_LABEL   = 0.01    # 레이블 상관계수 하한
CORR_FEAT    = 0.95    # 피처 간 상관계수 상한

COLUMNS = [
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes',
    'sttl','dttl','sloss','dloss','service','sload','dload','spkts','dpkts',
    'swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len',
    'sjit','djit','stime','ltime','sintpkt','dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login',
    'ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm',
    'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','label'
]

# 도메인 지식 기반 제거 피처 (Step 2)
DROP_DOMAIN = [
    'srcip','dstip',            # IP 주소 — 식별자
    'stime','ltime',            # 타임스탬프
    'stcpb','dtcpb',            # TCP 랜덤 시퀀스 번호
    'trans_depth','res_bdy_len',# HTTP 관련, 대부분 0
    'sport','dsport'            # 고카디널리티 포트 번호
]

# One-Hot Encoding 대상 피처 (Step 7)
OHE_COLS = ['proto', 'state', 'service']

# 로그 저장용
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)


# ============================================================
# 0. 데이터 로드
# ============================================================
def load_data(path):
    log("=" * 60)
    log("[LOAD] 데이터 로드")
    df = pd.read_csv(path, header=None, names=COLUMNS, low_memory=False)
    log(f"  원본: {df.shape[0]:,}행 × {df.shape[1]}열")

    # dsport: hex 문자열 → 정수
    df['dsport'] = df['dsport'].apply(
        lambda x: int(str(x), 16) if str(x).startswith('0x') else pd.to_numeric(x, errors='coerce')
    )
    # ct_ftp_cmd: 공백 문자열 → 정수
    df['ct_ftp_cmd'] = pd.to_numeric(
        df['ct_ftp_cmd'].astype(str).str.strip(), errors='coerce'
    ).fillna(0).astype(int)

    return df


# ============================================================
# Step 1. 중복 제거 + Worms 제거
# ============================================================
def step1_remove_duplicates_and_worms(df):
    log("\n[Step 1] 중복 제거 + Worms 제거")

    before = len(df)
    df = df.drop_duplicates()
    log(f"  중복 제거: {before:,} → {len(df):,}행 ({before - len(df):,}건 제거)")

    before = len(df)
    df = df[df['attack_cat'] != 'Worms']
    log(f"  Worms 제거: {before:,} → {len(df):,}행 ({before - len(df):,}건 제거)")

    n_normal = (df['label'] == 0).sum()
    n_attack = (df['label'] == 1).sum()
    log(f"  잔존: 정상 {n_normal:,} / 공격 {n_attack:,} / 불균형 {n_normal/n_attack:.2f}:1")

    return df.reset_index(drop=True)


# ============================================================
# Step 2. 도메인 지식 기반 피처 제거
# ============================================================
def step2_drop_domain_features(df):
    log("\n[Step 2] 도메인 지식 기반 피처 제거")
    log(f"  제거 대상: {DROP_DOMAIN}")
    df = df.drop(columns=DROP_DOMAIN)
    log(f"  잔존 피처: {df.shape[1]}개")
    return df


# ============================================================
# Step 3. 타입 변환 + 결측값 처리
# ============================================================
def step3_type_and_missing(df):
    log("\n[Step 3] 결측값 처리")

    df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].fillna(0)
    log(f"  ct_flw_http_mthd: NaN → 0")

    df['is_ftp_login'] = df['is_ftp_login'].fillna(0)
    log(f"  is_ftp_login: NaN → 0")

    # 잔존 결측 확인
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if remaining.empty:
        log("  잔존 결측값: 없음")
    else:
        log(f"  잔존 결측값:\n{remaining}")

    return df


# ============================================================
# Step 4. VarianceThreshold
# ============================================================
def step4_variance_threshold(df):
    log(f"\n[Step 4] VarianceThreshold (threshold={VAR_THRESH})")

    # attack_cat, label 제외하고 수치형만 대상
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['label']]

    variances = df[num_cols].var()
    low_var_cols = variances[variances < VAR_THRESH].index.tolist()

    log(f"  저분산 피처 ({len(low_var_cols)}개): {low_var_cols}")
    df = df.drop(columns=low_var_cols)
    log(f"  잔존 피처: {df.shape[1]}개")

    return df, low_var_cols


# ============================================================
# Step 5. 상관계수 기반 피처 제거
# ============================================================
def step5_correlation_filter(df):
    log(f"\n[Step 5] 상관계수 기반 피처 제거")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c != 'label']

    corr_matrix = df[num_cols].corr()

    # 5-1. 레이블과 |r| < CORR_LABEL → 제거
    label_corr = corr_matrix['label'].drop('label').abs()
    low_label_corr = label_corr[label_corr < CORR_LABEL].index.tolist()
    log(f"  레이블 무관 피처 (|r|<{CORR_LABEL}, {len(low_label_corr)}개): {low_label_corr}")

    # 5-2. 피처 간 |r| > CORR_FEAT → 레이블 상관 낮은 쪽 제거
    feat_corr = corr_matrix.loc[feature_cols, feature_cols]
    to_drop_high = set()
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            if abs(feat_corr.iloc[i, j]) > CORR_FEAT:
                f1, f2 = feature_cols[i], feature_cols[j]
                # 레이블 상관 낮은 피처 제거
                drop = f1 if label_corr.get(f1, 0) <= label_corr.get(f2, 0) else f2
                to_drop_high.add(drop)
    to_drop_high = list(to_drop_high)
    log(f"  고상관 피처 (|r|>{CORR_FEAT}, {len(to_drop_high)}개): {to_drop_high}")

    # 합산 제거
    all_drop = list(set(low_label_corr + to_drop_high))
    # 실제 존재하는 컬럼만 제거
    all_drop = [c for c in all_drop if c in df.columns]
    df = df.drop(columns=all_drop)
    log(f"  총 제거: {len(all_drop)}개")
    log(f"  잔존 피처: {df.shape[1]}개")

    return df, all_drop


# ============================================================
# Step 6. Train / Test 분리
# ============================================================
def step6_train_test_split(df):
    log("\n[Step 6] Train / Test 분리 (Stratified 8:2)")

    # attack_cat은 군집화 평가용으로 별도 보존
    y_cat = df['attack_cat']
    X = df.drop(columns=['label', 'attack_cat'])
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
        X, y, y_cat,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    log(f"  Train: {len(y_train):,}행 (정상 {(y_train==0).sum():,} / 공격 {(y_train==1).sum():,})")
    log(f"  Test:  {len(y_test):,}행  (정상 {(y_test==0).sum():,} / 공격 {(y_test==1).sum():,})")

    return X_train, X_test, y_train, y_test, y_cat_train, y_cat_test


# ============================================================
# Step 7. One-Hot Encoding
# ============================================================
def step7_one_hot_encoding(X_train, X_test):
    log("\n[Step 7] One-Hot Encoding")

    # 실제 존재하는 OHE 대상만 필터링
    ohe_cols = [c for c in OHE_COLS if c in X_train.columns]
    log(f"  인코딩 대상: {ohe_cols}")

    # Train fit
    X_train = pd.get_dummies(X_train, columns=ohe_cols, drop_first=False)
    train_cols = X_train.columns.tolist()

    # Test transform (Train 기준 컬럼 맞춤)
    X_test = pd.get_dummies(X_test, columns=ohe_cols, drop_first=False)
    X_test = X_test.reindex(columns=train_cols, fill_value=0)

    log(f"  인코딩 후 피처 수: {X_train.shape[1]}개")

    return X_train, X_test


# ============================================================
# Step 8. 언더샘플링 (Train만)
# ============================================================
def step8_undersample(X_train, y_train):
    log(f"\n[Step 8] Random Undersampling (정상:공격 = {US_RATIO}:1, Train만)")

    df_train = X_train.copy()
    df_train['label'] = y_train.values

    majority = df_train[df_train['label'] == 0]
    minority = df_train[df_train['label'] == 1]

    n_target = len(minority) * US_RATIO
    majority_down = resample(majority, replace=False,
                             n_samples=n_target, random_state=RANDOM_STATE)

    df_resampled = pd.concat([majority_down, minority]).sample(
        frac=1, random_state=RANDOM_STATE
    )

    X_train_res = df_resampled.drop(columns=['label'])
    y_train_res = df_resampled['label']

    log(f"  언더샘플링 전: 정상 {len(majority):,} / 공격 {len(minority):,}")
    log(f"  언더샘플링 후: 정상 {(y_train_res==0).sum():,} / 공격 {(y_train_res==1).sum():,}")

    return X_train_res, y_train_res


# ============================================================
# Step 9. RobustScaler
# ============================================================
def step9_scaling(X_train, X_test):
    log("\n[Step 9] RobustScaler (Train fit → Test transform)")

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    log(f"  스케일링 완료: Train {X_train_scaled.shape} / Test {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# 결과 저장
# ============================================================
def save_outputs(X_train, X_test, y_train, y_test, y_cat_train, y_cat_test):
    log("\n[SAVE] 결과 저장")

    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv',  index=False)
    y_train.to_csv('y_train.csv', index=False, header=True)
    y_test.to_csv('y_test.csv',   index=False, header=True)
    y_cat_train.to_csv('y_train_cat.csv', index=False, header=True)
    y_cat_test.to_csv('y_test_cat.csv',   index=False, header=True)

    log("  저장 완료:")
    for f in ['X_train.csv','X_test.csv','y_train.csv','y_test.csv',
              'y_train_cat.csv','y_test_cat.csv']:
        log(f"    {f}")

    # 로그 저장
    with open('preprocessing_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log("  preprocessing_report.txt")


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    df = load_data(FILE_PATH)

    df                              = step1_remove_duplicates_and_worms(df)
    df                              = step2_drop_domain_features(df)
    df                              = step3_type_and_missing(df)
    df, low_var_cols                = step4_variance_threshold(df)
    df, corr_drop_cols              = step5_correlation_filter(df)
    X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = step6_train_test_split(df)
    X_train, X_test                 = step7_one_hot_encoding(X_train, X_test)
    X_train, y_train                = step8_undersample(X_train, y_train)
    X_train, X_test, scaler         = step9_scaling(X_train, X_test)

    save_outputs(X_train, X_test, y_train, y_test, y_cat_train, y_cat_test)

    log("\n" + "=" * 60)
    log("전처리 완료")
    log(f"  X_train: {X_train.shape}")
    log(f"  X_test:  {X_test.shape}")
    log(f"  y_train: 정상 {(y_train==0).sum():,} / 공격 {(y_train==1).sum():,}")
    log(f"  y_test:  정상 {(y_test==0).sum():,} / 공격 {(y_test==1).sum():,}")
    log("=" * 60)
