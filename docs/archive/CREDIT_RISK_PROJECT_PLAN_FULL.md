Mình chốt project theo hướng này:

# Credit Risk Scoring & Loan Decisioning System

**Hệ thống chấm điểm rủi ro tín dụng và hỗ trợ quyết định cho vay**

Chỉ sử dụng **dataset Bài 12** làm nguồn dữ liệu chính. Mục tiêu không phải train thật nhiều model, mà xây một quy trình **Data Science end-to-end (khoa học dữ liệu từ đầu đến cuối)** đủ mạnh để đưa vào CV và đủ logic để bạn bảo vệ khi phỏng vấn.

---

# I. Sản phẩm cuối cùng phải làm được gì?

Khi nhập một hồ sơ khách hàng:

```text
Thu nhập
Tuổi
Nghề nghiệp
Thời gian làm việc
Khoản vay
Mục đích vay
Nợ hiện tại
Lịch sử tín dụng
Lịch sử chậm trả
...
```

hệ thống phải trả:

```text
Probability of Default – PD
(Xác suất vỡ nợ)
               ↓
13.8%

Internal Credit Score
(Điểm tín dụng nội bộ)
               ↓
672

Risk Grade
(Hạng rủi ro)
               ↓
C

Decision
(Quyết định)
               ↓
MANUAL REVIEW
(Thẩm định thêm)

Risk Drivers
(Các nguyên nhân chính tạo ra rủi ro)
               ↓
High Debt-to-Income
Previous Delinquency
High Loan-to-Income
```

Đây sẽ là output trung tâm của toàn bộ project.

---

# II. Phạm vi project

Project sẽ làm 10 khối chính:

1. Data Audit *(kiểm tra dữ liệu)*
2. EDA – Exploratory Data Analysis *(phân tích khám phá dữ liệu)*
3. Leakage Audit *(kiểm tra rò rỉ dữ liệu)*
4. Feature Engineering *(tạo đặc trưng)*
5. Credit Risk Modeling *(xây dựng mô hình rủi ro tín dụng)*
6. Probability Calibration *(hiệu chỉnh xác suất)*
7. Credit Scoring *(chấm điểm tín dụng)*
8. Decision Optimization *(tối ưu quyết định)*
9. Explainability & Fairness *(giải thích mô hình và kiểm tra công bằng)*
10. Deployment & Monitoring *(triển khai và giám sát)*

---

# GIAI ĐOẠN 0 — Chốt bài toán trước khi code

## Mục tiêu nghiệp vụ

Câu hỏi chính:

> Dựa trên những thông tin có tại thời điểm khách hàng xin vay, khách hàng có xác suất vỡ nợ bao nhiêu?

Target *(biến mục tiêu)*:

```text
loan_status

0 = không vỡ nợ
1 = vỡ nợ
```

Mô hình học:

[
PD=P(Default=1|X)
]

Không làm:

- dự đoán default trong 30/60/90 ngày;
- Early Warning System *(hệ thống cảnh báo sớm)*;
- transaction fraud *(gian lận giao dịch)*;
- AML – Anti-Money Laundering *(chống rửa tiền)*.

Vì dataset không có dữ liệu phù hợp.

### Deliverable *(sản phẩm đầu ra)*

Tạo:

```text
docs/problem_definition.md
```

Trong đó ghi rõ:

- mục tiêu;
- target;
- đối tượng sử dụng;
- input;
- output;
- giới hạn dữ liệu;
- giả định nghiệp vụ.

---

# GIAI ĐOẠN 1 — Thiết lập project

Repo nên có dạng:

```text
credit-risk-decisioning/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_analysis.ipynb
│   └── 04_model_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── validation.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── calibrate.py
│   │   └── scoring.py
│   │
│   ├── decision/
│   │   └── policy.py
│   │
│   ├── explainability/
│   │   └── shap_analysis.py
│   │
│   └── monitoring/
│       └── drift.py
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│
├── configs/
│   └── config.yaml
│
├── models/
├── reports/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── MODEL_CARD.md
```

### Công nghệ

Core *(phần cốt lõi)*:

```text
Python
Pandas
NumPy
Scikit-learn
Matplotlib
LightGBM
CatBoost
SHAP
```

Sản phẩm:

```text
FastAPI
Streamlit
Docker
```

Experiment Tracking *(theo dõi thí nghiệm)*:

```text
MLflow
```

MLflow có thể thêm sau khi pipeline chính chạy tốt.

---

# GIAI ĐOẠN 2 — Data Audit

Đây là bước đầu tiên thực sự làm với dataset.

## 2.1 Kiểm tra schema

Với từng cột:

- kiểu dữ liệu;
- min/max;
- missing;
- unique;
- distribution *(phân bố)*;
- business meaning *(ý nghĩa nghiệp vụ)*.

Tạo một **Data Dictionary (từ điển dữ liệu)** chuẩn.

---

## 2.2 Duplicate

Kiểm tra:

```text
Duplicate client_ID
Duplicate rows
```

Dataset hiện tại không có duplicate nhưng vẫn phải có validation tự động.

---

## 2.3 Missing Values

Hai vấn đề đã biết:

```text
person_emp_length
→ thiếu ~2.75%

loan_int_rate
→ thiếu ~9.56%
```

Không vội:

```text
fillna(mean)
```

Mà phải thử:

- median imputation *(điền bằng trung vị)*;
- group median *(trung vị theo nhóm)*;
- missing indicator *(biến đánh dấu dữ liệu bị thiếu)*.

Ví dụ:

```text
loan_int_rate_missing = 1
```

---

## 2.4 Outlier

Dataset có:

```text
person_age = 144
person_emp_length = 123
```

Xây **Business Rules (quy tắc nghiệp vụ)**.

Ví dụ:

```text
20 <= age <= 100
```

và:

```text
employment_length
<=
age - minimum_working_age
```

Không nhất thiết cứ outlier là xóa.

Phải so sánh:

```text
Giữ lại
vs
Clip
vs
Loại bỏ
```

---

## 2.5 Báo cáo Data Quality

Cuối bước này tạo:

```text
reports/data_quality_report.md
```

Có:

```text
Missing
Invalid
Outlier
Duplicate
Range violation
Potential leakage
```

### Hoàn thành khi

Có một pipeline có thể chạy:

```bash
python -m src.data.validation
```

và tự kiểm tra dataset.

---

# GIAI ĐOẠN 3 — EDA

Đây không phải vẽ 30 biểu đồ cho đẹp.

EDA phải trả lời câu hỏi nghiệp vụ.

---

## 3.1 Target Analysis

Phân bố:

```text
Non-default ≈ 78%
Default     ≈ 22%
```

Phân tích **Class Imbalance (mất cân bằng lớp)**.

Kết luận ban đầu:

> Có imbalance nhưng chưa đủ nghiêm trọng để mặc định dùng SMOTE.

---

# 3.2 Income và Default

Phân tích:

```text
Income distribution
Default rate by income band
```

Ví dụ chia:

```text
<25K
25–50K
50–75K
75–100K
>100K
```

Câu hỏi:

> Risk có giảm khi income tăng không?

---

# 3.3 Loan-to-Income

[
LTI=\frac{Loan}{Income}
]

Xem:

```text
LTI ↑
→ Default Rate ?
```

Đây rất có thể là một trong những feature quan trọng.

---

# 3.4 Debt-to-Income

[
DTI=\frac{Debt}{Income}
]

Xem:

```text
DTI ↑
→ Probability of Default ↑ ?
```

---

# 3.5 Previous Default

So sánh:

```text
Previous Default = YES
vs
Previous Default = NO
```

Default rate hiện tại khác nhau khá rõ.

---

# 3.6 Loan Intent

So sánh:

```text
Education
Medical
Personal
Debt Consolidation
Home Improvement
Venture
```

---

# 3.7 Home Ownership

So sánh:

```text
OWN
MORTGAGE
RENT
OTHER
```

---

# 3.8 Employment

Phân tích:

```text
Employment Type
Employment Length
```

với default.

---

# 3.9 Credit History

Phân tích:

```text
Credit History Length
Credit Utilization
Open Accounts
Past Delinquencies
Previous Default
```

---

# 3.10 Correlation / Redundancy

Đặc biệt:

```text
loan_percent_income
loan_to_income_ratio
```

đã có:

[
corr\approx0.999
]

Phải chọn một hoặc chứng minh lý do giữ cả hai.

---

# GIAI ĐOẠN 4 — Leakage Audit

Đây sẽ là một trong những phần mạnh nhất của project.

## Câu hỏi:

> Feature này tồn tại trước hay sau khi quyết định tín dụng được đưa ra?

Đặc biệt kiểm tra:

```text
loan_grade
loan_int_rate
```

---

## Tạo 2 bộ feature

### Primary Feature Set *(bộ biến chính)*

**Application-time Features (biến có tại thời điểm xin vay)**.

Đây là model chính.

Không sử dụng feature nào có nguy cơ được sinh ra bởi quy trình underwriting trước đó.

---

### Extended Feature Set *(bộ biến mở rộng)*

Có:

```text
loan_grade
loan_int_rate
```

nếu muốn nghiên cứu.

Sau đó thực hiện:

**Ablation Study (nghiên cứu loại bỏ/thêm từng nhóm biến)**.

Ví dụ:

| ModelFeaturesROC-AUC |                  |      |
| -------------------- | ---------------- | ---- |
| M1                   | Application-time | 0.xx |
| M2                   | + Interest Rate  | 0.xx |
| M3                   | + Loan Grade     | 0.xx |

Nếu `loan_grade` làm AUC nhảy mạnh:

> Đây là bằng chứng cần xem xét leakage.

Đừng lấy model AUC cao nhất bằng mọi giá.

---

# GIAI ĐOẠN 5 — Train / Validation / Test

Dataset không có thời gian.

Vì vậy **không giả tạo time split (chia dữ liệu theo thời gian)**.

Mình khuyên:

```text
80%
Development Set
(Tập phát triển)

20%
Locked Test Set
(Tập test khóa)
```

20% test:

> Tuyệt đối không dùng khi tuning.

Trong 80% còn lại:

```text
5-fold Stratified Cross Validation
(kiểm định chéo 5 phần giữ tỷ lệ lớp)
```

Tức là:

```text
Full Dataset
        │
      80/20
        │
 ┌──────┴─────────┐
 │                │
Development    Locked Test
80%              20%
 │
5-fold CV
```

Đây sạch hơn việc cứ nhìn test liên tục.

---

# GIAI ĐOẠN 6 — Feature Engineering

Không cần tạo 200 feature.

Chỉ tạo feature có ý nghĩa nghiệp vụ.

---

## 6.1 Financial burden

Ví dụ:

[
LoanToIncome=
\frac{LoanAmount}{Income}
]

[
DebtToIncome=
\frac{Debt}{Income}
]

[
TotalDebtExposure=
OtherDebt+LoanAmount
]

---

# 6.2 Employment Stability

Ví dụ:

[
EmploymentRatio=
\frac{EmploymentLength}{Age}
]

---

# 6.3 Credit History

Ví dụ:

```text
prior_credit_issue =
Previous Default
OR
Past Delinquencies > 0
```

---

# 6.4 Interaction Features

Có thể thử:

```text
High DTI × Previous Default

High LTI × Low Income

Rent × High Loan

Short Employment × High DTI
```

Nhưng chỉ giữ nếu cross-validation chứng minh có ích.

---

# GIAI ĐOẠN 7 — Baseline

Trước khi Boosting Model *(mô hình tăng cường)* phải có baseline.

### Model 0

**Dummy Classifier (mô hình giả cơ sở)**.

Mục đích:

> Model ML phải tốt hơn dự đoán ngây thơ.

---

### Model 1

**Logistic Regression (hồi quy logistic)**.

Đây rất quan trọng trong banking.

Nó:

- dễ giải thích;
- cho probability;
- là baseline mạnh;
- phù hợp credit scoring.

Đừng bỏ Logistic Regression chỉ vì LightGBM có AUC cao hơn.

---

# GIAI ĐOẠN 8 — Các model mạnh hơn

Mình không khuyên train 15 model.

Chỉ cần:

### Model A

Logistic Regression *(hồi quy logistic)*

### Model B

LightGBM *(mô hình cây tăng cường gradient hiệu quả)*

### Model C

CatBoost *(mô hình boosting hỗ trợ tốt biến phân loại)*

XGBoost có thể làm thêm nhưng không bắt buộc.

---

# GIAI ĐOẠN 9 — Model Evaluation

Không sử dụng Accuracy làm metric chính.

## Ranking Metrics *(chỉ số khả năng xếp hạng)*

### ROC-AUC

Khả năng xếp khách default cao hơn khách tốt.

### Gini

[
Gini=2AUC-1
]

### KS – Kolmogorov-Smirnov

Đo khoảng cách lớn nhất giữa phân bố score của:

```text
Good customers
vs
Bad customers
```

Rất phù hợp credit risk.

---

# Classification Metrics

Tại một threshold cụ thể:

```text
Precision
Recall
F1
Specificity
Confusion Matrix
```

---

# Imbalanced Metrics

Dùng:

**PR-AUC – Precision Recall Area Under Curve**
*(diện tích dưới đường Precision-Recall)*.

---

# GIAI ĐOẠN 10 — Hyperparameter Tuning

Chỉ tuning sau khi pipeline đã ổn.

Có thể dùng:

**Optuna (thư viện tối ưu siêu tham số)**.

Ví dụ LightGBM:

```text
num_leaves
max_depth
learning_rate
min_child_samples
subsample
colsample_bytree
regularization
```

Không tuning theo test.

Objective *(mục tiêu tối ưu)* có thể là:

```text
CV ROC-AUC
```

hoặc:

```text
CV PR-AUC
```

---

# GIAI ĐOẠN 11 — Probability Calibration

Đây là bước project Credit Risk rất nên có.

Giả sử model nói:

```text
PD = 20%
```

thì trong nhóm khách tương tự đó, ta muốn khoảng 20% thực sự default.

Đó gọi là:

**Calibration (hiệu chỉnh xác suất)**.

So sánh:

```text
Raw model
Platt Scaling
Isotonic Regression
```

Đánh giá bằng:

### Brier Score

[
BS=\frac1N\sum(p\_i-y\_i)^2
]

và:

**Calibration Curve (đường hiệu chỉnh xác suất)**.

---

# GIAI ĐOẠN 12 — Chọn Champion Model

Không chọn chỉ vì AUC cao nhất.

Tạo bảng:

| ModelAUCPR-AUCKSGiniBrierExplainability |   |   |   |   |   |            |
| --------------------------------------- | - | - | - | - | - | ---------- |
| Logistic                                |   |   |   |   |   | Cao        |
| LightGBM                                |   |   |   |   |   | Trung bình |
| CatBoost                                |   |   |   |   |   | Trung bình |

**Champion Model (mô hình chính thức được chọn)** phải cân bằng:

```text
Predictive performance
Khả năng dự đoán

Calibration
Chất lượng xác suất

Explainability
Khả năng giải thích

Stability
Độ ổn định
```

Không phải:

> AUC +0.002 → auto winner.

---

# GIAI ĐOẠN 13 — Credit Score

Đừng gọi là FICO Score.

Hãy gọi:

**Internal Credit Score (điểm tín dụng nội bộ)**.

Chuyển calibrated PD *(xác suất vỡ nợ đã hiệu chỉnh)* thành điểm.

Một cách tốt là sử dụng **log-odds transformation (biến đổi log của tỷ lệ odds)**:

[
Odds=\frac{1-PD}{PD}
]

sau đó:

[
Score=Offset+Factor\ln(Odds)
]

Ta định nghĩa rõ:

```text
Base Score
PDO – Points to Double the Odds
(Số điểm tăng khi odds tăng gấp đôi)
```

Ví dụ:

```text
PD 2%  → Score cao
PD 10% → Score trung bình
PD 40% → Score thấp
```

Điểm này là **internal risk score**, không giả vờ là điểm tín dụng chính thức của ngân hàng.

---

# GIAI ĐOẠN 14 — Risk Grade

Chuyển score/PD thành:

```text
A
B
C
D
E
```

Ví dụ:

```text
A → Very Low Risk
    Rủi ro rất thấp

B → Low Risk
    Rủi ro thấp

C → Medium Risk
    Rủi ro trung bình

D → High Risk
    Rủi ro cao

E → Very High Risk
    Rủi ro rất cao
```

Ngưỡng phải xác định từ dữ liệu, không đặt tùy tiện.

---

# GIAI ĐOẠN 15 — Decision Engine

Đây là bước biến ML thành Data Science.

Output:

```text
APPROVE
Duyệt

MANUAL REVIEW
Thẩm định thêm

REJECT
Từ chối
```

Có hai threshold:

[
T\_A
]

và:

[
T\_R
]

Ví dụ:

```text
PD < TA
→ APPROVE

TA ≤ PD < TR
→ REVIEW

PD ≥ TR
→ REJECT
```

---

# GIAI ĐOẠN 16 — Cost-sensitive Optimization

Dataset không cho ta lợi nhuận/thua lỗ thật.

Vì vậy xây một:

**Simulated Business Cost Matrix (ma trận chi phí nghiệp vụ mô phỏng)**.

Ví dụ:

```text
Approve good customer
→ lợi ích +1

Reject good customer
→ opportunity cost -0.2

Approve default customer
→ credit loss -5

Manual Review
→ operating cost -0.05
```

Những giá trị này phải ghi rõ:

> giả định mô phỏng cho portfolio.

Sau đó thử nhiều:

```text
TA
TR
```

và tối ưu:

```text
Expected Cost
(chi phí kỳ vọng)
```

hoặc:

```text
Expected Utility
(lợi ích kỳ vọng)
```

---

# GIAI ĐOẠN 17 — Đánh giá Decision Policy

Dashboard cuối không chỉ show AUC.

Phải có:

```text
Approval Rate
Tỷ lệ duyệt

Review Rate
Tỷ lệ thẩm định thêm

Rejection Rate
Tỷ lệ từ chối

Bad Rate Among Approved
Tỷ lệ default trong nhóm được duyệt

Captured Default Rate
Tỷ lệ khách xấu bị phát hiện

Expected Cost
Chi phí kỳ vọng
```

Đây là những metric cực hợp để kể câu chuyện business.

---

# GIAI ĐOẠN 18 — Explainable AI

Sử dụng:

**SHAP – SHapley Additive exPlanations**
*(phương pháp phân tích đóng góp của từng biến vào dự đoán)*.

---

## Global Explainability

*(giải thích toàn cục)*

Model học rằng feature nào quan trọng nhất?

Ví dụ:

```text
Loan-to-Income
Previous Default
DTI
Income
Home Ownership
...
```

Dùng:

```text
SHAP Summary Plot
SHAP Importance Plot
```

---

# Local Explainability

*(giải thích cho từng khách hàng)*

Ví dụ:

```text
Applicant: CUST_21983

PD: 36.7%

Risk increasing factors:

High DTI               +10.2%
Previous Default        +8.7%
High Loan-to-Income     +6.5%

Risk reducing factors:

Long Credit History     -3.1%
Stable Employment       -1.8%
```

Đây chính là nội dung nên hiện trên dashboard.

---

# GIAI ĐOẠN 19 — Fairness Audit

**Fairness Audit (kiểm tra tính công bằng)**.

Không phải mục tiêu là tuyên bố:

> Model hoàn toàn công bằng.

Mà là kiểm tra model có khác biệt lớn giữa các nhóm không.

Ví dụ theo gender:

```text
Male
Female
```

So sánh:

```text
ROC-AUC
Recall
False Positive Rate
Approval Rate
Average PD
Calibration
```

Có thể không cho `gender` vào model chính nhưng vẫn sử dụng nó để audit.

Đây là cách làm đẹp hơn.

---

# GIAI ĐOẠN 20 — Stress Testing

**Stress Testing (kiểm thử trong điều kiện bất lợi)**.

Ví dụ mô phỏng:

```text
Income giảm 20%
DTI tăng
Default history distribution thay đổi
```

Xem:

```text
Average PD
Approval Rate
Risk Grade Distribution
```

thay đổi thế nào.

Nhưng ghi rõ đây là:

> Simulated stress scenario *(kịch bản stress mô phỏng)*.

---

# GIAI ĐOẠN 21 — Model Monitoring

Do dataset không có timeline nên không được giả vờ production thật.

Ta xây:

**Monitoring Simulation (mô phỏng giám sát mô hình)**.

Theo dõi:

### Feature Drift

*(thay đổi phân bố feature)*

Ví dụ:

```text
Income distribution
DTI
LTI
Loan Amount
```

### Prediction Drift

*(thay đổi phân bố dự đoán)*

```text
PD distribution
Risk Grade distribution
```

### PSI – Population Stability Index

*(chỉ số ổn định quần thể)*.

### Business Drift

```text
Approval Rate
Review Rate
Reject Rate
```

Nếu sau này có labels thì mới đo:

```text
AUC Drift
KS Drift
Calibration Drift
```

---

# GIAI ĐOẠN 22 — FastAPI

Xây API tối thiểu.

### Endpoint 1

```text
POST /predict
```

Input:

```json
{
  "age": 31,
  "income": 65000,
  "loan_amount": 12000,
  "employment_length": 5,
  "previous_default": "N"
}
```

Output:

```json
{
  "pd": 0.083,
  "credit_score": 704,
  "risk_grade": "B",
  "decision": "APPROVE"
}
```

---

### Endpoint 2

```text
POST /explain
```

Trả về risk drivers.

### Endpoint 3

```text
GET /health
```

Để kiểm tra model/API.

Không cần 20 endpoint.

---

# GIAI ĐOẠN 23 — Dashboard

Dùng **Streamlit (framework tạo ứng dụng dữ liệu Python)**.

Dashboard nên có 4 trang.

## Trang 1 — Portfolio Overview

*(tổng quan danh mục)*

Hiển thị:

```text
Number of Applications
Default Rate
Average PD
Approval Rate
Risk Grade Distribution
```

---

## Trang 2 — Applicant Assessment

*(đánh giá từng hồ sơ)*

Form nhập thông tin khách hàng.

Output:

```text
PD
Credit Score
Risk Grade
Decision
```

---

# Trang 3 — Explanation

Hiển thị:

```text
Risk Drivers
SHAP
```

---

# Trang 4 — Model Monitoring

Hiển thị:

```text
PSI
Feature Drift
PD Distribution
Approval Rate Drift
```

Đừng biến project thành frontend project.

Dashboard đơn giản nhưng sạch.

---

# GIAI ĐOẠN 24 — Testing

Cần ít nhất:

### Data Tests

```text
schema
missing
range
duplicate
```

### Feature Tests

Ví dụ:

```text
LTI calculation
DTI calculation
```

### Model Tests

```text
0 <= PD <= 1
```

### Score Tests

```text
PD tăng
→ Credit Score phải giảm
```

### Decision Tests

```text
PD < TA
→ APPROVE
```

### API Tests

Input hợp lệ/không hợp lệ.

---

# GIAI ĐOẠN 25 — Reproducibility

**Reproducibility (khả năng tái lập kết quả)** là điểm mình muốn giữ.

Có:

```text
random_seed = 42
```

và:

```text
requirements.txt
config.yaml
```

Một command nên train được toàn bộ:

```bash
python -m src.models.train
```

Một command chạy dashboard:

```bash
streamlit run dashboard/app.py
```

Một command chạy API:

```bash
uvicorn api.main:app
```

---

# GIAI ĐOẠN 26 — MODEL\_CARD.md

**Model Card (tài liệu mô tả mô hình)** nên ghi:

```text
Purpose
Mục đích

Dataset
Dữ liệu

Target
Biến mục tiêu

Features
Các biến sử dụng

Excluded Features
Các biến bị loại

Training Strategy
Chiến lược huấn luyện

Evaluation
Đánh giá

Calibration
Hiệu chỉnh xác suất

Decision Policy
Chính sách quyết định

Limitations
Giới hạn

Fairness
Công bằng

Monitoring
Giám sát
```

Cái này rất đẹp khi recruiter mở GitHub.

---

# GIAI ĐOẠN 27 — README cho CV

README không nên dài 50 trang.

Đầu README phải trả lời ngay:

### Problem

> Dự đoán Probability of Default *(xác suất vỡ nợ)* từ thông tin hồ sơ vay.

### Dataset

```text
32,581 loan applications
29 variables
21.8% default
```

### Solution

```text
Raw Data
→ Data Validation
→ Feature Engineering
→ PD Model
→ Calibration
→ Credit Score
→ Decision Engine
→ Explainability
```

### Results

Ví dụ sau này điền:

```text
ROC-AUC
PR-AUC
KS
Gini
Brier Score
```

### Demo

Ảnh dashboard.

### Architecture

Một sơ đồ.

### Limitations

Trình bày thẳng:

- synthetic/enriched data;
- không có timeline;
- không có LGD/EAD thực tế;
- cost assumptions là simulation;
- không đại diện policy của ngân hàng thật.

Đây không làm project yếu đi.

Ngược lại, nó chứng minh bạn biết giới hạn dữ liệu.

---

# GIAI ĐOẠN 28 — Final Audit

Trước khi đưa CV phải audit toàn bộ.

## Data

```text
No leakage?
No impossible values?
Test untouched?
```

## Modeling

```text
Cross-validation?
Baseline comparison?
Calibration?
```

## Business

```text
Threshold có lý do?
Risk grades có rationale?
Cost assumptions được ghi rõ?
```

## Explainability

```text
Global?
Individual applicant?
```

## Engineering

```text
Clean repo?
Reproducible?
Tests pass?
API works?
Dashboard works?
```

## Documentation

```text
README?
Model Card?
Architecture?
Results?
Limitations?
```

---

# Thứ tự triển khai thực tế mình đề xuất

Đừng xây dashboard trước.

Đi đúng thứ tự:

```text
1. Problem Definition
        ↓
2. Data Audit
        ↓
3. EDA
        ↓
4. Leakage Audit
        ↓
5. Data Split
        ↓
6. Preprocessing
        ↓
7. Feature Engineering
        ↓
8. Logistic Regression Baseline
        ↓
9. LightGBM / CatBoost
        ↓
10. Model Comparison
        ↓
11. Hyperparameter Tuning
        ↓
12. Calibration
        ↓
13. Locked Test Evaluation
        ↓
14. Credit Score
        ↓
15. Risk Grade
        ↓
16. Decision Threshold Optimization
        ↓
17. SHAP
        ↓
18. Fairness Audit
        ↓
19. Stress / Drift Simulation
        ↓
20. FastAPI
        ↓
21. Streamlit Dashboard
        ↓
22. Testing
        ↓
23. README + Model Card
        ↓
24. Final Audit
```

---

# Các mốc nên freeze

Mình rất khuyên chia thành các **Milestone (cột mốc)**.

### M1 — Data Ready

Xong:

```text
Data audit
Data cleaning
EDA
Leakage analysis
Train/test split
```

Không sửa dataset lung tung sau mốc này.

---

### M2 — Modeling Ready

Xong:

```text
LR
LightGBM
CatBoost
Cross-validation
Metrics
```

---

### M3 — Risk Model Frozen

Xong:

```text
Champion model
Calibration
Locked test evaluation
```

Sau mốc này không tune lại trên test.

---

### M4 — Decision System Ready

Xong:

```text
PD
Credit Score
Risk Grade
Decision Policy
SHAP
```

---

### M5 — Portfolio Ready

Xong:

```text
API
Dashboard
Monitoring
Tests
README
Model Card
Docker
```

Sau M5 có thể đưa GitHub/CV.

---

# Những thứ KHÔNG nên thêm

Để tránh project bị bày vẽ, mình chủ động bỏ:

- Deep Learning *(học sâu)*;
- Transformer;
- LLM – Large Language Model *(mô hình ngôn ngữ lớn)*;
- Kafka;
- Spark;
- Kubernetes;
- Microservices *(kiến trúc vi dịch vụ)*;
- Cloud phức tạp;
- Graph Neural Network;
- chatbot.

Không cái nào cần để chứng minh bạn làm tốt credit risk với dataset này.

Một **Logistic Regression + LightGBM + calibration + SHAP + decision optimization** làm chuẩn còn giá trị hơn nhồi thêm 10 công nghệ.

---

# Tiêu chuẩn để mình coi project này “đủ mạnh đưa CV”

Không có con số AUC bắt buộc kiểu phải >0.95. Một project tốt không được đánh giá chỉ bằng metric.

Mình muốn cuối cùng project đạt đủ:

**Data Science**

```text
✓ Data quality audit
✓ Leakage audit
✓ Feature engineering
✓ Proper validation
✓ Multiple meaningful models
✓ Credit-risk metrics
✓ Calibration
```

**Banking**

```text
✓ Probability of Default
✓ Credit Score
✓ Risk Grade
✓ Decision Policy
✓ Cost-sensitive thresholds
✓ Explainability
```

**Model Risk**

```text
✓ Fairness audit
✓ Stress testing
✓ Drift monitoring
✓ Model Card
✓ Limitations
```

**Engineering**

```text
✓ Reproducible pipeline
✓ Tests
✓ API
✓ Dashboard
✓ Docker
```

**Portfolio**

```text
✓ Clean GitHub
✓ Strong README
✓ Architecture diagram
✓ Results table
✓ Demo screenshots
✓ CV-ready description
```

Nếu hoàn thành đúng toàn bộ scope này thì **một mình Bài 12 đã đủ tạo một flagship project rất mạnh cho hướng Data Scientist ngân hàng**. Phần quan trọng nhất không phải làm project thật “to”, mà là mỗi quyết định từ xử lý dữ liệu → model → PD → threshold → quyết định tín dụng đều có lý do và bảo vệ được khi interviewer hỏi.