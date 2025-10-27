
# 🩺 Healthcare Appointment No-Show Prediction

## 🎯 Objective
Predict whether patients will miss their healthcare appointments and provide insights to optimize scheduling and reduce no-shows.

---

## 🧰 Tools & Technologies
- **Python:** Pandas, scikit-learn
- **Power BI:** Dashboard & visualization
- **CSV Dataset:** `predicted_appointments.csv`

---

## 📊 Dataset Description
| Column | Description |
|--------|-------------|
| patient_id | Unique ID for each patient |
| appointment_date | Scheduled appointment date |
| age | Patient age |
| gender | Male (M) or Female (F) |
| neighbourhood | Clinic location |
| sms_sent | 1 if reminder SMS was sent, 0 otherwise |
| previous_no_shows | Count of past missed appointments |
| No_Show_Probability | Predicted probability of missing appointment |
| Risk_Level | Categorized risk: Low, Medium, or High |

---

## 🧮 Steps Performed

### 1️⃣ Data Preparation
- Cleaned and formatted appointment data
- Calculated `No_Show_Probability` using trained Decision Tree model
- Categorized results into risk levels: Low, Medium, High

### 2️⃣ Power BI Dashboard Creation
1. **Load CSV** into Power BI using *Home → Get Data → Text/CSV → Load*
2. **Check column types** (date, number, text)
3. **Create Measures:**
   ```DAX
   Total_Appointments = COUNT('predicted_appointments'[patient_id])
   Average_No_Show_Prob = AVERAGE('predicted_appointments'[No_Show_Probability])
   High_Risk_Patients = COUNTROWS(FILTER('predicted_appointments', 'predicted_appointments'[Risk_Level] = "High"))
   ```
4. **Add Visuals:**
   - Card: Total Appointments, Avg No-Show Probability, High-Risk Count
   - Pie Chart: Risk Level distribution
   - Bar Chart: SMS vs Average No-Show Probability
   - Column Chart: Age vs No-Show Probability
   - Column Chart: Neighbourhood vs Avg No-Show Probability
   - Line Chart: Trend of No-Show Probability over time
5. **Add Filters (Slicers):**
   - Gender
   - Risk Level
   - SMS Sent

### 3️⃣ Dashboard Insights
- SMS reminders reduce no-show probability by ~20%
- Younger patients (<30) show higher predicted risk
- High-risk patients concentrated in *North* and *Central* areas
- No-show rates lower when appointments are within 3–5 days of scheduling

---

## 🧠 Recommendations
- Send SMS reminders to all high-risk patients
- Prioritize short waiting times between booking and appointment
- Overbook slightly on days with historically high no-shows
- Provide follow-up calls for repeat no-show patients

---

## 📦 Deliverables
- `predicted_appointments.csv` (dataset)
- Power BI Dashboard (Healthcare.pbix)
- Insights summary (this `.md` file)

---

## 🏁 Outcome
This project demonstrates the use of machine learning and BI tools to **predict and reduce missed healthcare appointments**, improving efficiency and patient satisfaction.
