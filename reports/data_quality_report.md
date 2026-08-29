# Data Quality Report

## 1. Dataset Overview

- **Dataset:** Credit Risk Dataset
- **Rows (số dòng):** 32,581
- **Columns (số cột):** 29
- **Target (biến mục tiêu):** `loan_status`
- **Identifier (biến định danh):** `client_ID`
- **Target missing (số target bị thiếu):** 0
- **Default observations (số hồ sơ vỡ nợ):** 7,108
- **Default rate (tỷ lệ vỡ nợ):** 21.82%

> This report summarizes Data Audit findings only.
> Báo cáo này chỉ tổng hợp kết quả kiểm tra dữ liệu.
> Raw data has not been modified.
> Dữ liệu gốc chưa bị thay đổi.

---

## 2. Schema and Data Dictionary

### 2.1 Technical Schema Summary

| dtype | column_count |
| --- | --- |
| str | 12 |
| float64 | 9 |
| int64 | 8 |

### 2.2 Data Dictionary Coverage

- Dataset columns: 29
- Data Dictionary rows: 29
- Missing business meanings: 0

The Data Dictionary is used to document technical type,
business meaning, expected domain, and audit notes.

Data Dictionary (từ điển dữ liệu) dùng để mô tả kiểu dữ liệu,
ý nghĩa nghiệp vụ, miền giá trị kỳ vọng và ghi chú kiểm tra.

---

## 3. Missing Values

Missing values (dữ liệu thiếu) được ghi nhận ở bước Data Audit,
nhưng chưa được impute (điền dữ liệu thiếu) trực tiếp trên raw data.

| issue | affected_columns | affected_rows | affected_rate | severity | recommendation | status |
| --- | --- | --- | --- | --- | --- | --- |
| loan_int_rate có 3,116 giá trị thiếu (9.56%). | loan_int_rate | 3,116 | 9.56% | medium | Exclude from Primary Model until leakage audit; if retained, use median + missing indicator as baseline \| Cần kiểm chứng: First verify application-time availability in the leakage audit. If retained, compare imputation strategies inside train/CV. Do not use loan_grade for Primary Model imputation unless proven non-leaky. | open |
| person_emp_length có 895 giá trị thiếu (2.75%). | person_emp_length | 895 | 2.75% | medium | Median + missing indicator as baseline \| Cần kiểm chứng: Compare median, group median, and median + indicator inside train/CV only. Fit all imputation statistics on each training fold. | open |

---

## 4. Invalid Values

Invalid values (giá trị không hợp lệ) bao gồm các token hoặc
giá trị cần được xác minh trước khi chuẩn hóa thành missing.

| issue | affected_columns | affected_rows | affected_rate | severity | recommendation | status |
| --- | --- | --- | --- | --- | --- | --- |
| Không phát hiện encoded missing token (ký hiệu chuỗi đại diện dữ liệu thiếu) trong các cột text/categorical đã kiểm tra. | - | 0 | 0.00% | low | Duy trì kiểm tra này trong validation.py. | passed |

---

## 5. Duplicate Checks

Duplicate checks (kiểm tra dữ liệu trùng lặp) bao gồm duplicate rows
và duplicate `client_ID`.

| issue | affected_columns | affected_rows | affected_rate | severity | recommendation | status |
| --- | --- | --- | --- | --- | --- | --- |
| Duplicate rows (các dòng dữ liệu bị trùng hoàn toàn). | All columns | 0 | 0.00% | low | Không cần xử lý; tiếp tục validation tự động. | passed |
| Duplicate client_ID (mã khách hàng xuất hiện nhiều lần). | client_ID | 0 | 0.00% | low | client_ID đang unique (không trùng); duy trì validation. | passed |

---

## 6. Range Violations

Range violation (vi phạm khoảng giá trị nghiệp vụ) được xác định
bằng các business rules (quy tắc nghiệp vụ) đã khai báo.

| issue | affected_columns | affected_rows | affected_rate | severity | recommendation | status |
| --- | --- | --- | --- | --- | --- | --- |
| person_age nằm ngoài khoảng [20, 100]. | person_age | 5 | 0.02% | medium | Kiểm tra nguồn dữ liệu; nếu xác nhận là lỗi, ưu tiên loại các dòng vi phạm thay vì tự động clip (chặn giá trị về ngưỡng). | open |
| person_emp_length không hợp lý so với person_age theo business rule (quy tắc nghiệp vụ). | person_emp_length, person_age | 2 | 0.01% | medium | Nếu xác nhận person_emp_length là lỗi, đổi riêng giá trị sai thành NaN và xử lý trong preprocessing pipeline (quy trình tiền xử lý dữ liệu). | open |

---

## 7. Outliers and Business Rules

Outlier (giá trị ngoại lệ) không được tự động xóa hoặc clip
(chặn về ngưỡng) chỉ dựa trên thống kê.

Quyết định xử lý phải dựa trên business rule
(quy tắc nghiệp vụ) và mức độ hợp lý của dữ liệu.

| issue | affected_columns | affected_rows | affected_rate | severity | recommendation | status |
| --- | --- | --- | --- | --- | --- | --- |
| person_age outside [20, 100] | person_age | 5 | 0.02% | medium | Remove violating rows after review (loại các dòng vi phạm sau khi kiểm tra). | open |
| person_emp_length < 0 hoặc > person_age - 14 | person_age | 2 | 0.01% | medium | Set invalid person_emp_length to NaN, then impute inside train/CV pipeline (đổi riêng giá trị sai thành missing, sau đó điền trong quy trình train/kiểm định chéo). | open |

---

## 8. Potential Leakage Watchlist

Potential leakage (nguy cơ rò rỉ dữ liệu) là các biến có khả năng
chứa thông tin không phù hợp với thời điểm mô hình phải dự đoán.

Các biến trong nhóm này chưa bị xóa khỏi raw data.
Chúng cần được kiểm tra sâu ở Leakage Audit
(giai đoạn kiểm tra rò rỉ dữ liệu).

| issue | affected_columns | affected_rows | affected_rate | severity | recommendation | status |
| --- | --- | --- | --- | --- | --- | --- |
| Loan grade có thể là kết quả của quá trình đánh giá rủi ro tín dụng trước đó. Nếu được tạo sau khi hồ sơ đã được thẩm định, sử dụng biến này có thể làm model học lại kết quả chấm điểm cũ. | loan_grade | - | - | high | Không dùng mặc định trong Primary Model (mô hình chính) cho đến khi xác minh thời điểm biến được tạo. | deferred_to_leakage_audit |
| Lãi suất có thể được xác định trong hoặc sau quá trình thẩm định tín dụng. Nếu lãi suất được tạo dựa trên mức rủi ro đã đánh giá trước đó, biến này có thể chứa thông tin mà model không được phép biết tại thời điểm khách hàng nộp hồ sơ. | loan_int_rate | - | - | high | Không dùng mặc định trong Primary Model (mô hình chính) cho đến khi xác minh Application-time availability (biến có sẵn tại thời điểm khách hàng xin vay). | deferred_to_leakage_audit |

---

## 9. Recommended Cleaning and Validation Actions

Recommended actions (các hành động xử lý được đề xuất)
từ Data Audit:

| category | affected_columns | severity | recommendation | status |
| --- | --- | --- | --- | --- |
| Potential leakage | loan_grade | high | Không dùng mặc định trong Primary Model (mô hình chính) cho đến khi xác minh thời điểm biến được tạo. | deferred_to_leakage_audit |
| Potential leakage | loan_int_rate | high | Không dùng mặc định trong Primary Model (mô hình chính) cho đến khi xác minh Application-time availability (biến có sẵn tại thời điểm khách hàng xin vay). | deferred_to_leakage_audit |
| Missing | loan_int_rate | medium | Exclude from Primary Model until leakage audit; if retained, use median + missing indicator as baseline \| Cần kiểm chứng: First verify application-time availability in the leakage audit. If retained, compare imputation strategies inside train/CV. Do not use loan_grade for Primary Model imputation unless proven non-leaky. | open |
| Missing | person_emp_length | medium | Median + missing indicator as baseline \| Cần kiểm chứng: Compare median, group median, and median + indicator inside train/CV only. Fit all imputation statistics on each training fold. | open |
| Outlier | person_age | medium | Remove violating rows after review (loại các dòng vi phạm sau khi kiểm tra). | open |
| Outlier | person_age | medium | Set invalid person_emp_length to NaN, then impute inside train/CV pipeline (đổi riêng giá trị sai thành missing, sau đó điền trong quy trình train/kiểm định chéo). | open |
| Range violation | person_age | medium | Kiểm tra nguồn dữ liệu; nếu xác nhận là lỗi, ưu tiên loại các dòng vi phạm thay vì tự động clip (chặn giá trị về ngưỡng). | open |
| Range violation | person_emp_length, person_age | medium | Nếu xác nhận person_emp_length là lỗi, đổi riêng giá trị sai thành NaN và xử lý trong preprocessing pipeline (quy trình tiền xử lý dữ liệu). | open |

---

## 10. Checks to Implement in validation.py

Các kiểm tra sau nên được tự động hóa trong
`src/data/validation.py`:

| category | affected_columns | issue | severity |
| --- | --- | --- | --- |
| Missing | loan_int_rate | loan_int_rate có 3,116 giá trị thiếu (9.56%). | medium |
| Missing | person_emp_length | person_emp_length có 895 giá trị thiếu (2.75%). | medium |
| Range violation | person_age | person_age nằm ngoài khoảng [20, 100]. | medium |
| Range violation | person_emp_length, person_age | person_emp_length không hợp lý so với person_age theo business rule (quy tắc nghiệp vụ). | medium |
| Duplicate | All columns | Duplicate rows (các dòng dữ liệu bị trùng hoàn toàn). | low |
| Duplicate | client_ID | Duplicate client_ID (mã khách hàng xuất hiện nhiều lần). | low |
| Invalid | - | Không phát hiện encoded missing token (ký hiệu chuỗi đại diện dữ liệu thiếu) trong các cột text/categorical đã kiểm tra. | low |

---

## 11. Data Audit Conclusion

Data Audit đã kiểm tra các nhóm chính:

1. Missing values (dữ liệu thiếu).
2. Invalid values (giá trị không hợp lệ).
3. Outliers (giá trị ngoại lệ).
4. Duplicate records (bản ghi trùng lặp).
5. Range violations (vi phạm khoảng giá trị nghiệp vụ).
6. Potential leakage (nguy cơ rò rỉ dữ liệu).

Các quyết định về imputation (điền dữ liệu thiếu),
outlier treatment (xử lý giá trị ngoại lệ) và feature exclusion
(loại biến khỏi mô hình) chưa được áp dụng trực tiếp trên
`df_raw`.

Những quyết định này sẽ được triển khai ở các giai đoạn
Preprocessing (tiền xử lý dữ liệu), Leakage Audit
(kiểm tra rò rỉ dữ liệu) và Modeling
(xây dựng mô hình) sau khi có validation phù hợp.
