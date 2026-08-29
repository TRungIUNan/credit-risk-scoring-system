# Credit Risk Scoring & Loan Decisioning System
## Tổng quan sản phẩm và giải thích Dataset Bài 12

---

# 1. Tên project

**Credit Risk Scoring & Loan Decisioning System**  
**Hệ thống chấm điểm rủi ro tín dụng và hỗ trợ quyết định cho vay**

---

# 2. Mục tiêu của project

Project xây dựng một hệ thống hỗ trợ ngân hàng đánh giá rủi ro của một khách hàng tại thời điểm khách hàng nộp hồ sơ vay.

Từ các thông tin về:

- đặc điểm khách hàng;
- thu nhập và việc làm;
- khoản vay;
- mức nợ hiện tại;
- lịch sử tín dụng;
- hành vi tín dụng trong quá khứ;

hệ thống sẽ ước lượng:

**Probability of Default - PD (xác suất vỡ nợ)**

\[
PD = P(Default = 1 \mid X)
\]

Sau đó chuyển kết quả dự đoán thành:

```text
Thông tin hồ sơ
        ↓
Credit Risk Model
(Mô hình rủi ro tín dụng)
        ↓
Probability of Default - PD
(Xác suất vỡ nợ)
        ↓
Internal Credit Score
(Điểm tín dụng nội bộ)
        ↓
Risk Grade
(Hạng rủi ro)
        ↓
Decision
(Quyết định)
        ↓
APPROVE / MANUAL REVIEW / REJECT
Duyệt / Thẩm định thêm / Từ chối
```

Project tập trung vào:

**Application Credit Risk / Loan Underwriting**  
**Đánh giá rủi ro tín dụng tại thời điểm khách hàng xin vay**

Không coi dataset này là đủ để xây:

- **Early Warning System (hệ thống cảnh báo nợ xấu sớm)**;
- **Fraud Detection (phát hiện gian lận)**;
- **AML - Anti-Money Laundering (chống rửa tiền)**;
- mô hình dự đoán default trong 30/60/90 ngày;

vì dataset không có chuỗi thời gian hành vi khách hàng đủ cho các bài toán đó.

---

# 3. Tổng quan Dataset Bài 12

Dataset có:

- **32,581 hồ sơ vay**
- **29 cột**
- Target: `loan_status`
- `loan_status = 0`: khách hàng không vỡ nợ
- `loan_status = 1`: khách hàng vỡ nợ
- Default rate khoảng **21.8%**

Các cột có thể chia thành 8 nhóm:

1. Định danh khách hàng
2. Nhân khẩu học
3. Thu nhập và việc làm
4. Thông tin khoản vay
5. Gánh nặng tài chính
6. Lịch sử tín dụng
7. Địa lý
8. Target

---

# 4. Toàn bộ 29 feature trong dataset

## 4.1. `client_ID`

### Ý nghĩa
Mã định danh duy nhất của một hồ sơ/khách hàng trong dataset.

Ví dụ:

```text
CUST_00001
CUST_00002
...
```

### Kiểu dữ liệu
**Identifier (biến định danh)**

### Có thể dùng để làm gì?
- Theo dõi một hồ sơ.
- Join kết quả prediction với dữ liệu gốc.
- Hiển thị Applicant ID trên dashboard.
- Truy vết prediction trong log.

### Có nên dùng train model?
**Không.**

ID không mang ý nghĩa rủi ro tín dụng thực sự. Nếu model học từ ID thì gần như chỉ học nhiễu hoặc cấu trúc nhân tạo của dataset.

### Vai trò trong project

```text
Tracking / Logging / Dashboard
```

---

# 5. Nhóm nhân khẩu học

## 5.1. `person_age`

### Ý nghĩa
Tuổi của người vay.

### Kiểu dữ liệu
**Numerical Feature (biến số)**

### Có thể phân tích gì?
- Default rate theo nhóm tuổi.
- Khách hàng trẻ có rủi ro cao hơn hay thấp hơn?
- Tuổi có tương tác với credit history hay employment length không?

Có thể chia:

```text
20–24
25–34
35–44
45–54
55+
```

### Có thể tạo feature gì?

Ví dụ:

```text
age_band
```

hoặc:

\[
EmploymentRatio =
\frac{EmploymentLength}{Age}
\]

\[
CreditHistoryRatio =
\frac{CreditHistoryLength}{Age}
\]

### Cảnh báo Data Quality
Dataset có tuổi bất thường, từng phát hiện giá trị lên tới khoảng:

```text
144 tuổi
```

Do đó cần kiểm tra business rule, ví dụ:

```text
20 <= person_age <= 100
```

---

## 5.2. `gender`

### Ý nghĩa
Giới tính của khách hàng.

### Kiểu dữ liệu
**Categorical Feature (biến phân loại)**

### Có thể dùng để làm gì?
Phù hợp nhất cho:

**Fairness Audit (kiểm tra tính công bằng của mô hình)**

Ví dụ so sánh giữa các nhóm:

- Approval Rate - tỷ lệ được duyệt
- Recall
- False Positive Rate
- Average PD
- Calibration

### Có nên dùng train model?
Project chính nên cân nhắc **không dùng `gender` làm input trực tiếp**, nhưng vẫn giữ biến này cho fairness audit.

Lý do: đây là thuộc tính nhạy cảm trong quyết định tín dụng và cần thận trọng về tính công bằng cũng như bối cảnh pháp lý.

### Vai trò đề xuất

```text
Không dùng trong Primary Credit Model
→ Giữ lại cho Fairness Audit
```

---

## 5.3. `marital_status`

### Ý nghĩa
Tình trạng hôn nhân của khách hàng.

Ví dụ:

- Single
- Married
- Divorced
- Widowed

### Kiểu dữ liệu
**Categorical Feature**

### Có thể phân tích gì?
- Default rate theo tình trạng hôn nhân.
- Tương tác với income, debt hoặc home ownership.

### Có nên dùng model?
Có thể thử trong model, nhưng chỉ nên giữ nếu:
- có predictive value;
- có business rationale;
- không tạo vấn đề fairness đáng kể.

Không nên mặc định cho rằng một tình trạng hôn nhân nào đó "rủi ro hơn" nếu chưa được dữ liệu chứng minh.

---

## 5.4. `education_level`

### Ý nghĩa
Trình độ học vấn của người vay.

Ví dụ:

- High School
- Bachelor
- Master
- PhD

### Kiểu dữ liệu
**Categorical Feature**

### Có thể phân tích gì?
- Education level và income.
- Education level và employment type.
- Default rate theo education level.

### Có thể dùng model?
Có thể thử.

Tuy nhiên cần kiểm tra:
- biến này có thêm giá trị ngoài income/employment hay không;
- có tạo ra chênh lệch không hợp lý giữa các nhóm hay không.

---

# 6. Nhóm nhà ở, thu nhập và việc làm

## 6.1. `person_home_ownership`

### Ý nghĩa
Hình thức sở hữu nhà ở hiện tại của khách hàng.

Các giá trị gồm:

- RENT - thuê nhà
- OWN - sở hữu nhà
- MORTGAGE - đang trả khoản vay thế chấp
- OTHER - khác

### Kiểu dữ liệu
**Categorical Feature**

### Ý nghĩa nghiệp vụ
Home ownership có thể phản ánh một phần:
- mức độ ổn định tài chính;
- nghĩa vụ tài chính hiện có;
- tài sản và khả năng chi trả.

### Có thể phân tích gì?
Default rate giữa:

```text
RENT
OWN
MORTGAGE
OTHER
```

### Có thể tạo interaction gì?

Ví dụ:

```text
RENT × High DTI
RENT × High Loan-to-Income
MORTGAGE × Other Debt
```

### Có thể dùng model?
**Có**, đây là feature khá hợp với bài toán credit risk.

---

## 6.2. `person_income`

### Ý nghĩa
Thu nhập của khách hàng, trong dataset được hiểu là thu nhập hàng năm.

### Kiểu dữ liệu
**Numerical Feature**

### Ý nghĩa nghiệp vụ
Thu nhập là một trong những yếu tố chính để đánh giá:

**Repayment Capacity (khả năng trả nợ)**

### Có thể phân tích gì?
- Default rate theo income band.
- Thu nhập của default vs non-default.
- Income kết hợp với loan amount.
- Income kết hợp với debt.

Có thể chia thành các band:

```text
< 25K
25K–50K
50K–75K
75K–100K
> 100K
```

### Có thể tạo feature gì?

\[
LoanToIncome =
\frac{LoanAmount}{Income}
\]

\[
DebtToIncome =
\frac{Debt}{Income}
\]

### Cảnh báo
Income thường có phân bố lệch phải và có outlier lớn.

Có thể cần:
- log transformation;
- winsorization;
- robust scaling;

tùy model và kết quả EDA.

---

## 6.3. `person_emp_length`

### Ý nghĩa
Số năm khách hàng đã làm việc.

### Kiểu dữ liệu
**Numerical Feature**

### Ý nghĩa nghiệp vụ
Có thể phản ánh:

**Employment Stability (độ ổn định việc làm)**

### Có thể phân tích gì?
- Employment length càng dài thì risk có giảm không?
- Người mới đi làm có default nhiều hơn không?

Có thể chia:

```text
< 1 năm
1–3 năm
3–5 năm
5–10 năm
> 10 năm
```

### Có thể tạo feature gì?

\[
EmploymentRatio =
\frac{EmploymentLength}{Age}
\]

### Cảnh báo Data Quality
Dataset có:
- missing values;
- một số giá trị rất lớn;
- có trường hợp employment length không hợp lý so với tuổi.

Cần business rule kiểu:

```text
person_emp_length
<= person_age - minimum_working_age
```

---

## 6.4. `employment_type`

### Ý nghĩa
Loại hình việc làm của khách hàng.

Ví dụ:

- Full-time
- Part-time
- Self-employed
- Unemployed

### Kiểu dữ liệu
**Categorical Feature**

### Ý nghĩa nghiệp vụ
Có thể phản ánh:
- tính ổn định thu nhập;
- khả năng trả nợ đều đặn;
- độ biến động của nguồn thu.

### Có thể phân tích gì?
Default rate theo employment type.

### Có thể tạo interaction gì?

```text
Unemployed × High DTI
Self-employed × Income
Part-time × High LTI
```

### Có thể dùng model?
**Có.**

---

# 7. Nhóm thông tin khoản vay

## 7.1. `loan_intent`

### Ý nghĩa
Mục đích sử dụng khoản vay.

Dataset gồm các nhóm như:

- EDUCATION - giáo dục
- MEDICAL - y tế
- VENTURE - kinh doanh/đầu tư
- PERSONAL - chi tiêu cá nhân
- DEBTCONSOLIDATION - hợp nhất nợ
- HOMEIMPROVEMENT - sửa chữa/nâng cấp nhà

### Kiểu dữ liệu
**Categorical Feature**

### Có thể phân tích gì?
- Default rate theo loan intent.
- Loan amount trung bình theo từng mục đích.
- Income/DTI theo từng mục đích vay.

### Ý nghĩa cho project
Có thể trả lời câu hỏi:

> Mục đích vay nào có hồ sơ rủi ro cao hơn trong dataset?

### Có thể dùng model?
**Có.**

---

## 7.2. `loan_amnt`

### Ý nghĩa
Số tiền khách hàng yêu cầu vay.

### Kiểu dữ liệu
**Numerical Feature**

### Ý nghĩa nghiệp vụ
Loan amount riêng lẻ chưa đủ để biết risk.

Ví dụ:

```text
Khoản vay 20,000
```

có thể:
- nhỏ đối với người thu nhập 200,000;
- rất lớn đối với người thu nhập 25,000.

Do đó cần kết hợp với income và debt.

### Có thể tạo feature gì?

\[
LoanToIncome =
\frac{LoanAmount}{Income}
\]

\[
TotalDebtExposure =
OtherDebt + LoanAmount
\]

Có thể thêm:

\[
LoanPerMonth =
\frac{LoanAmount}{LoanTermMonths}
\]

Đây không phải monthly payment thật vì chưa tính lãi suất, nhưng có thể dùng như một proxy nếu được ghi rõ.

### Có thể dùng model?
**Có.**

---

## 7.3. `loan_int_rate`

### Ý nghĩa
Lãi suất áp dụng cho khoản vay.

### Kiểu dữ liệu
**Numerical Feature**

### Ý nghĩa nghiệp vụ
Lãi suất cao có thể:
- làm tăng burden của khoản vay;
- hoặc phản ánh rằng hệ thống underwriting trước đó đã đánh giá khách hàng là rủi ro.

### Có thể phân tích gì?
- Interest rate vs default.
- Interest rate theo loan grade.
- Interest rate theo DTI/LTI.

### Cảnh báo quan trọng: Leakage
Cần xác định:

> Lãi suất có tồn tại trước khi model ra quyết định hay được ngân hàng xác định sau khi đánh giá risk?

Nếu lãi suất đã được pricing system tạo ra dựa trên risk thì sử dụng nó trong application model có thể gây:

**Data Leakage (rò rỉ dữ liệu)**

### Vai trò đề xuất
Không mặc định dùng trong Primary Model.

Thực hiện:

```text
Model A: Không loan_int_rate
Model B: Có loan_int_rate
→ So sánh
```

### Data Quality
Cột này có missing values tương đối đáng kể và cần xử lý riêng.

---

## 7.4. `loan_term_months`

### Ý nghĩa
Thời hạn khoản vay tính theo tháng.

### Kiểu dữ liệu
**Numerical / Ordinal Feature**

### Ý nghĩa nghiệp vụ
Cùng một loan amount:

```text
12 tháng
vs
60 tháng
```

sẽ tạo nghĩa vụ thanh toán khác nhau.

### Có thể tạo feature gì?

\[
LoanAmountPerMonthProxy =
\frac{LoanAmount}{LoanTermMonths}
\]

hoặc interaction:

```text
Loan Amount × Loan Term
Interest Rate × Loan Term
LTI × Loan Term
```

### Có thể dùng model?
**Có.**

---

## 7.5. `loan_grade`

### Ý nghĩa
Hạng tín dụng của khoản vay.

Các mức:

```text
A
B
C
D
E
F
G
```

Thông thường:
- A đại diện risk thấp hơn;
- G đại diện risk cao hơn.

### Kiểu dữ liệu
**Ordinal Categorical Feature (biến phân loại có thứ tự)**

### Vì sao feature này rất mạnh?
Trong dataset, default rate tăng rất mạnh khi grade xấu đi.

Điều này cho thấy `loan_grade` đã tổng hợp rất nhiều tín hiệu risk.

### Cảnh báo cực kỳ quan trọng: Leakage
Nếu loan grade được tạo bởi một hệ thống credit scoring trước khi dataset được ghi nhận, thì dùng nó để xây một credit scoring model mới sẽ tạo vòng lặp:

```text
Risk Features
        ↓
Existing Risk Assessment
        ↓
loan_grade
        ↓
New Model
        ↓
Predict Risk
```

Model mới lúc này có thể đang "học lại kết quả chấm điểm cũ".

### Vai trò đề xuất
`loan_grade` phải được coi là:

**Potential Leakage Feature (feature có nguy cơ rò rỉ)**

Primary Model nên chạy không có `loan_grade`.

Sau đó thực hiện Ablation Study:

```text
Application-time Model
vs
Application-time + loan_grade
```

Nếu metric tăng mạnh, cần giải thích rõ thay vì chỉ chọn model có metric cao nhất.

---

# 8. Nhóm gánh nặng tài chính

## 8.1. `loan_percent_income`

### Ý nghĩa
Tỷ lệ khoản vay so với thu nhập.

Về logic gần tương đương:

\[
\frac{LoanAmount}{Income}
\]

### Kiểu dữ liệu
**Numerical Ratio Feature**

### Ý nghĩa nghiệp vụ
Cho biết quy mô khoản vay lớn đến mức nào so với khả năng kiếm tiền của khách hàng.

### Có thể phân tích gì?
Default rate theo các nhóm:

```text
Low LTI
Medium LTI
High LTI
Very High LTI
```

### Cảnh báo
Feature này gần như trùng với:

```text
loan_to_income_ratio
```

Correlation đã kiểm tra gần:

```text
0.999
```

Không nên giữ cả hai một cách vô thức.

---

## 8.2. `loan_to_income_ratio`

### Ý nghĩa
Tỷ lệ số tiền vay trên thu nhập.

\[
LTI =
\frac{LoanAmount}{Income}
\]

### Kiểu dữ liệu
**Numerical Ratio Feature**

### Ý nghĩa nghiệp vụ
Đây là một trong những feature quan trọng nhất cho:

**Affordability Assessment (đánh giá khả năng đáp ứng khoản vay)**

### Ví dụ

```text
Income = 50,000
Loan Amount = 10,000

LTI = 10,000 / 50,000 = 0.20
```

### Có thể dùng model?
**Có**, nhưng nên chọn một trong:

```text
loan_to_income_ratio
loan_percent_income
```

hoặc tự tính lại từ raw columns để đảm bảo nhất quán.

---

## 8.3. `other_debt`

### Ý nghĩa
Các khoản nợ khác mà khách hàng đang có ngoài khoản vay đang xin.

Theo Data Dictionary của dataset, đây là một biến được mô phỏng/enriched.

### Kiểu dữ liệu
**Numerical Feature**

### Ý nghĩa nghiệp vụ
Cho biết khách hàng đang có bao nhiêu debt exposure trước khi nhận khoản vay mới.

### Có thể tạo feature gì?

\[
TotalDebtExposure =
OtherDebt + LoanAmount
\]

Có thể kết hợp:

\[
DebtBurdenRatio =
\frac{OtherDebt + LoanAmount}{Income}
\]

### Cảnh báo
Có thể có outlier lớn và cần kiểm tra distribution.

---

## 8.4. `debt_to_income_ratio`

### Ý nghĩa
Tỷ lệ nợ so với thu nhập.

\[
DTI =
\frac{Debt}{Income}
\]

### Kiểu dữ liệu
**Numerical Ratio Feature**

### Ý nghĩa nghiệp vụ
Đây là một trong những biến rất quan trọng trong credit risk vì nó đo:

**Debt Burden (gánh nặng nợ)**

DTI cao có nghĩa một phần lớn năng lực tài chính của khách hàng đang bị sử dụng cho các nghĩa vụ nợ.

### Có thể phân tích gì?
- Default rate theo DTI.
- DTI kết hợp với previous default.
- DTI kết hợp với employment type.
- DTI kết hợp với LTI.

### Có thể tạo interaction gì?

```text
High DTI × Previous Default
High DTI × Short Employment
High DTI × High LTI
```

### Có thể dùng model?
**Có, đây là feature cốt lõi.**

---

# 9. Nhóm lịch sử tín dụng

## 9.1. `cb_person_default_on_file`

### Ý nghĩa
Cho biết khách hàng đã từng có default trong lịch sử tín dụng hay chưa.

Ví dụ:

```text
Y = từng default
N = chưa từng default
```

### Kiểu dữ liệu
**Binary Categorical Feature (biến phân loại nhị phân)**

### Ý nghĩa nghiệp vụ
Một khách hàng từng có default trong quá khứ có thể có risk cao hơn ở khoản vay mới.

### Có thể tạo feature gì?

```text
previous_default_flag = 1/0
```

Có thể kết hợp với `past_delinquencies`:

```text
prior_credit_issue =
previous_default == Y
OR
past_delinquencies > 0
```

### Có thể dùng model?
**Có, rất phù hợp.**

---

## 9.2. `cb_person_cred_hist_length`

### Ý nghĩa
Số năm lịch sử tín dụng của khách hàng.

### Kiểu dữ liệu
**Numerical Feature**

### Ý nghĩa nghiệp vụ
Lịch sử tín dụng càng dài thường giúp hệ thống có nhiều thông tin hơn để đánh giá khách hàng.

Khách hàng có lịch sử rất ngắn thường được gọi là:

**Thin-file Customer (khách hàng có hồ sơ tín dụng mỏng)**

### Có thể tạo feature gì?

```text
credit_history_band
```

Ví dụ:

```text
<3 năm
3–5 năm
6–10 năm
>10 năm
```

Hoặc:

\[
CreditHistoryAgeRatio =
\frac{CreditHistoryLength}{Age}
\]

### Có thể dùng model?
**Có.**

---

## 9.3. `credit_utilization_ratio`

### Ý nghĩa
Tỷ lệ sử dụng tín dụng.

Thông thường được hiểu gần dạng:

\[
CreditUtilization =
\frac{UsedCredit}{AvailableCreditLimit}
\]

### Kiểu dữ liệu
**Numerical Ratio Feature**

### Ý nghĩa nghiệp vụ
Tỷ lệ quá cao có thể cho thấy khách hàng đang sử dụng phần lớn hạn mức tín dụng hiện có.

### Có thể phân tích gì?
Default rate theo:

```text
<30%
30–50%
50–70%
70–90%
>90%
```

### Có thể tạo interaction gì?

```text
High Utilization × Past Delinquencies
High Utilization × High DTI
```

### Lưu ý
Nếu phân tích đơn biến không thấy khác biệt rõ giữa default/non-default, không nên ép kết luận feature này quan trọng.

Tree-based model vẫn có thể khai thác interaction của nó với feature khác.

---

## 9.4. `open_accounts`

### Ý nghĩa
Số tài khoản tín dụng đang mở của khách hàng.

### Kiểu dữ liệu
**Discrete Numerical Feature (biến số rời rạc)**

### Ý nghĩa nghiệp vụ
Số tài khoản quá nhiều có thể phản ánh:
- mức sử dụng tín dụng cao;
- nhiều nguồn nghĩa vụ tài chính;

nhưng một người có nhiều tài khoản cũng có thể chỉ đơn giản là có lịch sử tín dụng lâu.

Do đó cần xem kết hợp với:
- credit history length;
- utilization;
- income;
- debt.

### Có thể tạo feature gì?

Ví dụ:

\[
AccountsPerCreditYear =
\frac{OpenAccounts}{CreditHistoryLength}
\]

nếu feature này có ý nghĩa qua validation.

---

## 9.5. `past_delinquencies`

### Ý nghĩa
Số lần khách hàng từng thanh toán trễ hoặc có delinquency trong quá khứ.

### Kiểu dữ liệu
**Discrete Numerical Feature**

### Ý nghĩa nghiệp vụ
Delinquency là một tín hiệu hành vi tín dụng quan trọng.

### Có thể tạo feature gì?

```text
has_delinquency =
1 if past_delinquencies > 0
else 0
```

hoặc band:

```text
0
1
2+
```

Kết hợp:

```text
past_delinquencies × previous_default
past_delinquencies × utilization
```

### Có thể dùng model?
**Có.**

---

# 10. Nhóm địa lý

## 10.1. `country`

### Ý nghĩa
Quốc gia của khách hàng.

Dataset gồm chủ yếu:

- USA
- UK
- Canada

### Kiểu dữ liệu
**Categorical Feature**

### Có thể phân tích gì?
- Default rate theo quốc gia.
- Income distribution theo quốc gia.
- Loan amount theo quốc gia.

### Có nên dùng model?
Có thể benchmark, nhưng không nên mặc định giữ.

Nếu default rate gần như không khác giữa các nước, feature này có thể không mang nhiều predictive value.

Ngoài ra cần suy nghĩ về:
- policy differences;
- fairness;
- khả năng generalize.

---

## 10.2. `state`

### Ý nghĩa
Bang, tỉnh hoặc vùng hành chính của khách hàng.

### Kiểu dữ liệu
**High-cardinality Categorical Feature (biến phân loại có nhiều giá trị)**

### Có thể phân tích gì?
- Phân bố số hồ sơ theo region.
- Default rate theo region khi sample đủ lớn.

### Cảnh báo
Một số state có ít hồ sơ, dẫn đến default rate không ổn định.

Nếu sử dụng:
- không nên one-hot hàng loạt một cách vô thức;
- cần kiểm tra sample size;
- có thể nhóm rare category.

### Có nên dùng Primary Model?
Không phải feature bắt buộc.

---

## 10.3. `city`

### Ý nghĩa
Thành phố của khách hàng.

### Kiểu dữ liệu
**High-cardinality Categorical Feature**

### Có thể dùng để làm gì?
- Phân tích geographic distribution.
- Dashboard/map nếu cần.
- Kiểm tra concentration của portfolio.

### Cảnh báo
City có thể:
- cardinality cao;
- gây overfitting;
- vô tình encode socioeconomic status/geography.

### Vai trò đề xuất
Ưu tiên cho EDA/dashboard hơn là Primary Model, trừ khi validation chứng minh rõ giá trị.

---

## 10.4. `city_latitude`

### Ý nghĩa
Vĩ độ của thành phố.

### Kiểu dữ liệu
**Numerical Geographic Feature**

### Có thể dùng để làm gì?
- Vẽ bản đồ.
- Geographic visualization.
- Kiểm tra dữ liệu location.

### Có nên dùng model?
Không ưu tiên.

Latitude không có ý nghĩa credit risk trực tiếp.

Nếu model học được signal từ latitude, nó có thể chỉ đang học geographic proxy.

---

## 10.5. `city_longitude`

### Ý nghĩa
Kinh độ của thành phố.

### Kiểu dữ liệu
**Numerical Geographic Feature**

### Có thể dùng để làm gì?
- Map visualization.
- Geographic analysis.
- Data validation.

### Có nên dùng model?
Tương tự `city_latitude`, không ưu tiên cho Primary Model.

---

# 11. Target

## 11.1. `loan_status`

### Ý nghĩa
Biến mục tiêu cho biết khoản vay có default hay không.

```text
0 = Non-default
1 = Default
```

### Kiểu dữ liệu
**Binary Target (biến mục tiêu nhị phân)**

### Mô hình cần học

Không chỉ học:

```text
Default / Non-default
```

mà cần tạo:

\[
PD = P(loan\_status=1 \mid X)
\]

### Vì sao cần probability?
Một output:

```text
0
```

không cho biết khách hàng an toàn đến mức nào.

Trong khi:

```text
PD = 2%
PD = 18%
PD = 47%
```

cho phép xây:
- risk grade;
- credit score;
- decision threshold;
- business policy.

---

# 12. Tóm tắt vai trò của từng feature

| Feature | Nhóm | Vai trò đề xuất |
|---|---|---|
| `client_ID` | ID | Không train; tracking |
| `person_age` | Demographic | Train sau cleaning |
| `gender` | Sensitive attribute | Fairness audit; cân nhắc loại khỏi model |
| `marital_status` | Demographic | Candidate |
| `education_level` | Demographic | Candidate |
| `person_home_ownership` | Financial stability | Train |
| `person_income` | Repayment capacity | Core feature |
| `person_emp_length` | Employment stability | Core sau cleaning/imputation |
| `employment_type` | Employment stability | Train |
| `loan_intent` | Loan | Train |
| `loan_amnt` | Loan | Core feature |
| `loan_int_rate` | Loan/Pricing | Potential leakage; ablation |
| `loan_term_months` | Loan | Train |
| `loan_grade` | Risk grade | High leakage risk; không dùng Primary Model |
| `loan_percent_income` | Affordability | Redundant với LTI; chọn một |
| `loan_to_income_ratio` | Affordability | Core feature; chọn thay `loan_percent_income` |
| `other_debt` | Debt burden | Train |
| `debt_to_income_ratio` | Debt burden | Core feature |
| `cb_person_default_on_file` | Credit history | Core feature |
| `cb_person_cred_hist_length` | Credit history | Train |
| `credit_utilization_ratio` | Credit behavior | Candidate |
| `open_accounts` | Credit exposure | Candidate |
| `past_delinquencies` | Credit behavior | Core feature |
| `country` | Geography | Candidate / audit |
| `state` | Geography | Optional |
| `city` | Geography | EDA/dashboard; optional model |
| `city_latitude` | Geography | Map/EDA; không ưu tiên train |
| `city_longitude` | Geography | Map/EDA; không ưu tiên train |
| `loan_status` | Target | Target, không phải input |

---

# 13. Feature set đề xuất cho model chính

Primary Model nên ưu tiên các biến có thể biết tại thời điểm application.

## Core Application Features

```text
person_age
person_home_ownership
person_income
person_emp_length
employment_type

loan_intent
loan_amnt
loan_term_months

other_debt
debt_to_income_ratio
loan_to_income_ratio

cb_person_default_on_file
cb_person_cred_hist_length
credit_utilization_ratio
open_accounts
past_delinquencies
```

Có thể thử thêm:

```text
marital_status
education_level
country
```

nhưng chỉ giữ nếu validation chứng minh có ích.

---

# 14. Feature không nên đưa thẳng vào Primary Model

## Không train

```text
client_ID
```

## Sensitive / Audit

```text
gender
```

## Potential Leakage

```text
loan_grade
loan_int_rate
```

## Redundant

Chọn một trong:

```text
loan_percent_income
loan_to_income_ratio
```

## Geographic Proxy / High Cardinality

Cân nhắc loại khỏi Primary Model:

```text
state
city
city_latitude
city_longitude
```

---

# 15. Feature Engineering có thể tạo từ dataset

Dataset đủ để tạo thêm các feature có ý nghĩa mà không cần nguồn dữ liệu ngoài.

## 15.1. Loan-to-Income

\[
LTI =
\frac{LoanAmount}{Income}
\]

Đo khoản vay lớn đến mức nào so với thu nhập.

---

## 15.2. Debt-to-Income

Sử dụng trực tiếp feature có sẵn hoặc kiểm tra/tính lại nếu đủ dữ liệu.

\[
DTI =
\frac{Debt}{Income}
\]

Đo gánh nặng nợ so với thu nhập.

---

## 15.3. Total Debt Exposure

\[
TotalDebtExposure =
OtherDebt + LoanAmount
\]

Ước lượng tổng exposure sau khi khoản vay mới được cấp.

---

## 15.4. Total Debt Exposure Ratio

\[
TotalDebtExposureRatio =
\frac{OtherDebt + LoanAmount}{Income}
\]

Đánh giá khoản vay mới trong bối cảnh nợ hiện tại.

---

## 15.5. Employment Stability Ratio

\[
EmploymentStability =
\frac{EmploymentLength}{Age}
\]

Không mặc định tốt hơn raw feature; phải validate.

---

## 15.6. Credit History Ratio

\[
CreditHistoryRatio =
\frac{CreditHistoryLength}{Age}
\]

Đánh giá lịch sử tín dụng tương đối so với độ tuổi.

---

## 15.7. Prior Credit Issue

```text
prior_credit_issue = 1
nếu:
previous default = Yes
hoặc past delinquencies > 0
```

---

## 15.8. Thin-file Flag

```text
thin_file = 1
nếu credit history quá ngắn
```

Ngưỡng phải được chọn từ distribution/business rationale.

---

## 15.9. High Utilization Flag

```text
high_utilization = 1
nếu credit utilization vượt threshold nghiên cứu
```

Threshold không được coi là policy thật nếu chỉ đặt theo giả định.

---

## 15.10. Interaction Features

Có thể nghiên cứu:

```text
High DTI × Previous Default
High LTI × Low Income
High LTI × Rent
Short Employment × High DTI
High Utilization × Delinquency
Previous Default × Delinquency
```

Tree models có thể tự học nhiều interaction, vì vậy không cần tạo quá nhiều feature thủ công.

---

# 16. Các câu hỏi Data Science có thể trả lời bằng dataset

Dataset cho phép nghiên cứu:

1. Những yếu tố nào liên quan mạnh nhất tới default?
2. Loan-to-Income tăng thì default risk thay đổi thế nào?
3. Debt-to-Income ảnh hưởng ra sao?
4. Previous default có làm tăng PD đáng kể không?
5. Past delinquency ảnh hưởng như thế nào?
6. Credit history dài có liên quan đến risk thấp hơn không?
7. Người RENT, OWN và MORTGAGE khác nhau thế nào?
8. Loan intent nào có default rate cao?
9. Employment stability ảnh hưởng như thế nào?
10. Logistic Regression và boosting model khác nhau ra sao?
11. `loan_grade` có tạo leakage không?
12. Model có calibration tốt không?
13. Model có chênh lệch đáng kể giữa các demographic group không?
14. Threshold nào tạo trade-off phù hợp giữa approval và credit loss?

---

# 17. Các output chính của sản phẩm

Sau khi hoàn thiện, một hồ sơ có thể trả:

```text
Applicant ID:
CUST_09382

Probability of Default:
6.4%

Internal Credit Score:
728

Risk Grade:
B - Low Risk

Decision:
APPROVE
```

Kèm giải thích:

```text
Risk Increasing Factors:
+ High Debt-to-Income
+ Past Delinquency
+ High Loan-to-Income

Risk Reducing Factors:
- Long Credit History
- Stable Employment
```

---

# 18. Các mô hình dự kiến

## Baseline

- Dummy Classifier
- Logistic Regression

## Candidate Models

- LightGBM
- CatBoost
- Optional: XGBoost

Không thêm Deep Learning chỉ để project trông phức tạp hơn.

---

# 19. Evaluation

## Ranking Metrics

- ROC-AUC
- PR-AUC
- KS - Kolmogorov-Smirnov
- Gini

\[
Gini = 2 \times AUC - 1
\]

## Classification Metrics

- Precision
- Recall
- F1
- Specificity
- Confusion Matrix

## Probability Metrics

- Brier Score
- Calibration Curve

## Business Metrics

- Approval Rate
- Review Rate
- Rejection Rate
- Bad Rate Among Approved
- Captured Default Rate
- Expected Cost
- Expected Utility

---

# 20. Probability Calibration

Model không chỉ cần ranking tốt.

Nếu model đưa:

```text
PD = 10%
```

thì nhóm khách hàng được dự đoán quanh mức đó nên có tỷ lệ default quan sát tương đối gần 10%.

So sánh:

- Raw probability
- Platt Scaling
- Isotonic Regression

Đánh giá bằng:

- Brier Score
- Calibration Curve

---

# 21. Internal Credit Score

Sau khi có calibrated PD:

\[
Odds =
\frac{1-PD}{PD}
\]

\[
Score =
Offset + Factor \times \ln(Odds)
\]

Yêu cầu quan trọng:

```text
PD tăng
→ Credit Score giảm
```

Đây là **Internal Credit Score (điểm tín dụng nội bộ của project)**, không phải FICO Score.

---

# 22. Risk Grade

PD/score được chuyển thành:

```text
A - Very Low Risk
B - Low Risk
C - Medium Risk
D - High Risk
E - Very High Risk
```

Ngưỡng không được đặt tùy ý.

Cần dựa trên:
- PD distribution;
- observed default rate;
- business cost simulation.

---

# 23. Decision Engine

Sử dụng hai threshold:

```text
PD < T_APPROVE
→ APPROVE

T_APPROVE <= PD < T_REJECT
→ MANUAL REVIEW

PD >= T_REJECT
→ REJECT
```

Threshold phải được chọn thông qua:

**Cost-sensitive Optimization (tối ưu có xét chi phí nghiệp vụ)**

---

# 24. Explainability

Dùng:

**SHAP - SHapley Additive exPlanations**  
**Phương pháp giải thích đóng góp của từng feature vào prediction**

## Global Explainability

Trả lời:

> Toàn bộ model đang dựa vào feature nào nhiều nhất?

## Local Explainability

Trả lời:

> Tại sao hồ sơ cụ thể này bị đánh PD cao?

---

# 25. Fairness Audit

Các biến demographic như `gender` có thể dùng để kiểm tra:

- Approval Rate
- Recall
- False Positive Rate
- Average PD
- Calibration

Không tuyên bố model hoàn toàn công bằng chỉ dựa trên một chỉ số.

---

# 26. Monitoring Simulation

Vì dataset không có timeline thật, monitoring chỉ là mô phỏng.

Có thể theo dõi:

## Feature Drift
- Income
- DTI
- LTI
- Loan Amount

## Prediction Drift
- PD distribution
- Credit Score distribution
- Risk Grade distribution

## Population Stability
- PSI - Population Stability Index

## Business Drift
- Approval Rate
- Review Rate
- Rejection Rate

---

# 27. Giới hạn quan trọng của dataset

Project phải trình bày rõ:

1. Dataset có các thuộc tính simulated/enriched.
2. Không phải dữ liệu production thật của ngân hàng.
3. Không có application date.
4. Không có payment history theo thời gian.
5. Không có transaction history đầy đủ.
6. Không đủ để làm Early Warning System đúng nghĩa.
7. Không có LGD - Loss Given Default.
8. Không có EAD - Exposure at Default.
9. `loan_grade` có nguy cơ leakage.
10. `loan_int_rate` cần kiểm tra point-in-time availability.
11. Business cost chỉ có thể mô phỏng.
12. Monitoring chỉ là simulated monitoring.
13. Internal Credit Score không phải điểm tín dụng chính thức.
14. Decision policy không đại diện chính sách của một ngân hàng thực.

---

# 28. Pipeline sản phẩm cuối cùng

```text
Credit Risk Dataset
        ↓
Data Quality Audit
        ↓
EDA
        ↓
Leakage Audit
        ↓
Application-time Feature Set
        ↓
Feature Engineering
        ↓
Development / Locked Test Split
        ↓
Cross Validation
        ↓
Logistic Regression
LightGBM
CatBoost
        ↓
Champion Model
        ↓
Probability Calibration
        ↓
Calibrated PD
        ↓
Internal Credit Score
        ↓
Risk Grade
        ↓
Cost-based Decision Policy
        ↓
APPROVE / REVIEW / REJECT
        ↓
SHAP Explanation
        ↓
Fairness Audit
        ↓
Stress & Drift Simulation
        ↓
FastAPI
        ↓
Streamlit Dashboard
```

---

# 29. Definition of Done

## Data
- [ ] Giải thích đủ toàn bộ feature
- [ ] Data quality audit
- [ ] Missing handling
- [ ] Outlier handling
- [ ] Leakage audit
- [ ] Redundancy analysis

## Modeling
- [ ] Dummy baseline
- [ ] Logistic Regression
- [ ] LightGBM
- [ ] CatBoost
- [ ] Cross-validation
- [ ] Hyperparameter tuning
- [ ] Locked test evaluation
- [ ] Calibration

## Credit Risk
- [ ] Probability of Default
- [ ] KS
- [ ] Gini
- [ ] Internal Credit Score
- [ ] Risk Grade
- [ ] Decision Policy
- [ ] Business threshold optimization

## Model Risk
- [ ] SHAP global
- [ ] SHAP local
- [ ] Fairness audit
- [ ] Stress testing
- [ ] Monitoring simulation
- [ ] Model Card

## Engineering
- [ ] Reproducible pipeline
- [ ] Automated tests
- [ ] FastAPI
- [ ] Streamlit
- [ ] Docker

## Portfolio
- [ ] Clean README
- [ ] Architecture diagram
- [ ] Results table
- [ ] Demo screenshots
- [ ] Limitations
- [ ] CV-ready description

---

# 30. Kết luận

Một mình Dataset Bài 12 đã chứa đủ các nhóm thông tin quan trọng để xây dựng một project **Application Credit Risk** hoàn chỉnh:

```text
Customer Profile
        +
Income & Employment
        +
Loan Characteristics
        +
Debt Burden
        +
Credit History
        ↓
Probability of Default
        ↓
Credit Score
        ↓
Risk Grade
        ↓
Loan Decision
```

Giá trị lớn nhất của project không nằm ở việc dùng thật nhiều model hay thật nhiều dataset, mà ở việc chứng minh được toàn bộ tư duy:

```text
Dữ liệu nào được dùng?
Vì sao được dùng?
Có tồn tại tại thời điểm ra quyết định không?
Feature nào có leakage?
Model được đánh giá như thế nào?
Probability có đáng tin không?
Prediction được biến thành quyết định ra sao?
Quyết định có giải thích được không?
Model có hạn chế gì?
```

Đây là nền tảng để project đủ mạnh trở thành một **flagship Data Science project định hướng ngân hàng**.
