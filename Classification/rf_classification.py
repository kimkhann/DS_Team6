"""
UNSW-NB15 Random Forest Classification Pipeline
==============================================

사용법:
    python rf_classification.py

입력 파일:
    X_train.csv
    X_test.csv
    y_train.csv
    y_test.csv

출력 파일:
    rf_classification_report.txt

Plot 출력:
    Classification Metrics Bar Chart
    Confusion Matrix Heatmap
    Top 20 Feature Importance Plot
    Label Distribution Plot
    Classification Report Heatmap
    ROC Curve
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# ============================================================
# 설정
# ============================================================

RANDOM_STATE = 42

# GridSearchCV로 찾은 최적 파라미터
N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 5
N_JOBS = -1

# 현재 파이썬 파일 기준 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X_TRAIN_PATH = os.path.join(BASE_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(BASE_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(BASE_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(BASE_DIR, "y_test.csv")

# 로그 저장용 리스트
log_lines = []

# ============================================================
# 로그 출력 함수
# ============================================================

def log(msg):
    """
    화면 출력 + txt 저장용 로그 함수
    """
    print(msg)
    log_lines.append(msg)

# ============================================================
# Step 1. 전처리된 데이터 로드
# ============================================================

def step1_load_preprocessed_data():

    log("=" * 60)
    log("[Step 1] 전처리된 CSV 데이터 로드")

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    log(f"  X_train shape: {X_train.shape}")
    log(f"  X_test shape: {X_test.shape}")
    log(f"  y_train shape: {y_train.shape}")
    log(f"  y_test shape: {y_test.shape}")

    log(f"  Train label 분포: 정상 {(y_train == 0).sum():,} / 공격 {(y_train == 1).sum():,}")
    log(f"  Test label 분포: 정상 {(y_test == 0).sum():,} / 공격 {(y_test == 1).sum():,}")

    return X_train, X_test, y_train, y_test

# ============================================================
# Step 2. 5-Fold CV 기반 평균 F1 평가
# ============================================================

def step2_cv_evaluate(X_train, y_train):
    """
    Stratified 5-Fold CV 기반 평균 F1-score 계산
    """

    log("\n[Step 2-1] 5-Fold CV 평가")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    scores = cross_val_score(
        rf,
        X_train,
        y_train,
        cv=cv,
        scoring='f1'
    )



    log(f"Fold별 F1:\n{scores}")
    log(f"최고 F1: {scores.mean():.16f}")
    log(f"표준편차: {scores.std():.4f}")

    return scores.mean()

# ============================================================
# Step 3. Random Forest 모델 생성
# ============================================================

def step3_create_model():

    log("\n[Step 3] Random Forest 모델 생성")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    log("  모델 파라미터 설정 완료")
    log(f"  n_estimators: {N_ESTIMATORS}")
    log(f"  max_depth: {MAX_DEPTH}")
    log(f"  min_samples_leaf: {MIN_SAMPLES_LEAF}")
    log(f"  min_samples_split: {MIN_SAMPLES_SPLIT}")
    log("  max_features: sqrt")

    return model

# ============================================================
# Step 2-1. Threshold Validation
# ============================================================

# def step2_threshold_validation(X_train, y_train):

#     log("\n[Step 2-2] Threshold Validation")

#     thresholds = np.arange(0.50, 1.00, 0.05)

#     cv = StratifiedKFold(
#         n_splits=5,
#         shuffle=True,
#         random_state=RANDOM_STATE
#     )

#     results = []

#     for threshold in thresholds:

#         fold_f1_scores = []

#         for train_idx, valid_idx in cv.split(X_train, y_train):

#             X_fold_train = X_train.iloc[train_idx]
#             X_fold_valid = X_train.iloc[valid_idx]

#             y_fold_train = y_train.iloc[train_idx]
#             y_fold_valid = y_train.iloc[valid_idx]

#             rf = RandomForestClassifier(
#                 n_estimators=N_ESTIMATORS,
#                 max_depth=MAX_DEPTH,
#                 max_features='sqrt',
#                 random_state=RANDOM_STATE,
#                 n_jobs=N_JOBS
#             )

#             rf.fit(X_fold_train, y_fold_train)

#             y_prob = rf.predict_proba(X_fold_valid)[:, 1]

#             y_pred = (y_prob >= threshold).astype(int)

#             fold_f1 = f1_score(y_fold_valid, y_pred)

#             fold_f1_scores.append(fold_f1)

#         mean_f1 = np.mean(fold_f1_scores)
#         std_f1 = np.std(fold_f1_scores)

#         results.append({
#             "threshold": threshold,
#             "mean_f1": mean_f1,
#             "std_f1": std_f1
#         })

#         log(
#             f"Threshold={threshold:.2f} | "
#             f"Mean F1={mean_f1:.4f} | "
#             f"Std={std_f1:.4f}"
#         )

#     results_df = pd.DataFrame(results)

#     best_row = results_df.loc[
#         results_df["mean_f1"].idxmax()
#     ]

#     best_threshold = best_row["threshold"]

#     log("\n[Best Threshold]")
#     log(
#         f"Threshold={best_threshold:.2f} "
#         f"| Mean F1={best_row['mean_f1']:.4f}"
#     )

#     plt.figure(figsize=(8, 5))

#     plt.plot(
#         results_df["threshold"],
#         results_df["mean_f1"],
#         marker='o'
#     )

#     plt.xlabel("Threshold")
#     plt.ylabel("Mean F1-score")
#     plt.title("Threshold Validation")

#     plt.grid()
#     plt.show()

#     return best_threshold, results_df

# ============================================================
# Step 4. 모델 학습
# ============================================================

def step4_train_model(model, X_train, y_train):

    log("\n[Step 4] Random Forest 모델 학습")

    model.fit(X_train, y_train)

    log("  모델 학습 완료")

    return model

# ============================================================
# Step 5. Threshold 적용 예측
# ============================================================

def step5_predict(model, X_test, threshold=0.5):

    log(f"\n[Step 5] threshold={threshold}")

    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= threshold).astype(int)

    log(f"예측 완료: {len(y_pred):,}")

    return y_pred

# ============================================================
# Step 6. 분류 성능 평가
# ============================================================

def step6_evaluate_model(y_test, y_pred):

    log("\n[Step 6] 모델 성능 평가")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    log(f"  Accuracy : {accuracy:.4f}")
    log(f"  Precision: {precision:.4f}")
    log(f"  Recall : {recall:.4f}")
    log(f"  F1-score : {f1:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        digits=4
    )

    log("\n[Classification Report]")
    log(report)

    return accuracy, precision, recall, f1, report

# ============================================================
# Step 7. Confusion Matrix 생성
# ============================================================

def step7_confusion_matrix(y_test, y_pred):

    log("\n[Step 7] Confusion Matrix 생성")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Normal", "Actual_Attack"],
        columns=["Pred_Normal", "Pred_Attack"]
    )

    log(str(cm_df))

    return cm_df

# ============================================================
# Step 8. Feature Importance 추출
# ============================================================

def step8_feature_importance(model, X_train):

    log("\n[Step 8] Feature Importance 추출")

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    log("\n[Top 20 Feature Importance]")
    log(str(importance_df.head(20)))

    return importance_df

# ============================================================
# Step 9. 결과 파일 저장
# ============================================================

def step9_save_results(report):

    log("\n결과 저장")

    report_path = os.path.join(BASE_DIR, "rf_classification_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        f.write("\n\n")
        f.write(report)

    log(f"저장 완료: {report_path}")

# ============================================================
# Step 10. 결과 시각화
# ============================================================

def step10_visualize_results(accuracy, precision, recall, f1, cm_df, importance_df):

    print("\n[Step 10] 결과 시각화")

    # 1. 평가 지표 막대그래프
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    scores = [accuracy, precision, recall, f1]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, scores)

    plt.ylim(0, 1.05)
    plt.title("Random Forest Classification Metrics")
    plt.ylabel("Score")

    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.4f}", ha="center")

    plt.show()

    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))

    plt.imshow(cm_df.values)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks([0, 1], ["Pred_Normal", "Pred_Attack"])
    plt.yticks([0, 1], ["Actual_Normal", "Actual_Attack"])

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):

            value = cm_df.values[i, j]
            color = "black" if value > cm_df.values.max() / 2 else "white"

            plt.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color=color
            )

    plt.colorbar()
    plt.show()

    # 3. Top 20 Feature Importance
    top20 = importance_df.head(20).sort_values("importance")

    plt.figure(figsize=(10, 8))

    plt.barh(top20["feature"], top20["importance"])

    plt.title("Top 20 Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.show()

# ============================================================
# Step 11. Label Distribution 시각화
# ============================================================

def step11_plot_label_distribution(y_train, y_test):

    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    labels = ["Normal", "Attack"]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))

    plt.bar(x - width / 2, train_counts.values, width, label="Train")
    plt.bar(x + width / 2, test_counts.values, width, label="Test")

    plt.xticks(x, labels)

    plt.ylabel("Count")
    plt.title("Label Distribution")

    plt.legend()
    plt.grid(axis="y")

    plt.show()

# ============================================================
# Step 12. Classification Report Heatmap
# ============================================================

def step12_plot_classification_report(y_test, y_pred):

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_plot = report_df.loc[
        ["Normal", "Attack", "macro avg", "weighted avg"],
        ["precision", "recall", "f1-score"]
    ]

    plt.figure(figsize=(8, 5))

    plt.imshow(report_plot.values)

    plt.xticks(
        range(len(report_plot.columns)),
        report_plot.columns
    )

    plt.yticks(
        range(len(report_plot.index)),
        report_plot.index
    )

    for i in range(report_plot.shape[0]):
        for j in range(report_plot.shape[1]):

            plt.text(
                j,
                i,
                f"{report_plot.values[i, j]:.4f}",
                ha="center",
                va="center"
            )

    plt.title("Classification Report Heatmap")

    plt.colorbar()
    plt.tight_layout()

    plt.show()

# ============================================================
# Step 13. ROC Curve 시각화
# ============================================================

def step13_plot_roc_curve(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))

    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()
    plt.grid()

    plt.show()

    print(f"ROC AUC: {auc_score:.4f}")

# ============================================================
# 메인 실행부
# ============================================================

if __name__ == "__main__":

    # Step 1. 전처리된 데이터 불러오기
    X_train, X_test, y_train, y_test = step1_load_preprocessed_data()

    # Step 2. 5-Fold CV 평균 F1 평가
    cv_f1 = step2_cv_evaluate(
        X_train,
        y_train
    )

    # Step 3. Random Forest 모델 생성
    rf_model = step3_create_model()

    # Step 4. Train 데이터로 모델 학습
    rf_model = step4_train_model(
        rf_model,
        X_train,
        y_train
    )

    # Step 5. Test 데이터 예측
    y_pred = step5_predict(
        rf_model,
        X_test,
        threshold=0.5
    )

    # Step 6. 성능 평가
    accuracy, precision, recall, f1, report = step6_evaluate_model(
        y_test,
        y_pred
    )

    # Step 7. Confusion Matrix 생성
    cm_df = step7_confusion_matrix(
        y_test,
        y_pred
    )

    # Step 8. Feature Importance 추출
    importance_df = step8_feature_importance(
        rf_model,
        X_train
    )

    log("\n" + "=" * 60)
    log("Random Forest 분류 완료")
    log(f"5-Fold CV F1: {cv_f1:.4f}")
    log("Threshold: 0.5")
    log(f"Accuracy : {accuracy:.4f}")
    log(f"Precision: {precision:.4f}")
    log(f"Recall : {recall:.4f}")
    log(f"F1-score : {f1:.4f}")
    log("=" * 60)

    # Step 10. 결과 시각화
    step10_visualize_results(
        accuracy,
        precision,
        recall,
        f1,
        cm_df,
        importance_df
    )

    step11_plot_label_distribution(
        y_train,
        y_test
    )

    step12_plot_classification_report(
        y_test,
        y_pred
    )

    step13_plot_roc_curve(
        rf_model,
        X_test,
        y_test
    )

    # 결과 저장
    step9_save_results(
        report
    )