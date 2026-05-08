import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Set page config
st.set_page_config(page_title="INX Future Inc. Employee Data Dashboard", layout="wide")

# Title
st.title("INX Future Inc. Employee Data Dashboard 📈")


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('inx_data.csv')
    return df

@st.cache_resource
def load_model():
    search_paths = [
        Path(__file__).resolve().parent,
        Path.cwd(),
    ]
    if sys.argv:
        try:
            search_paths.append(Path(sys.argv[0]).resolve().parent)
        except Exception:
            pass

    for directory in search_paths:
        if directory is None:
            continue
        for model_name in ['xgb_model_pipeline.pkl', 'xgb_model_pipeline.joblib']:
            model_path = directory / model_name
            if model_path.exists():
                try:
                    return joblib.load(model_path), model_path, None
                except Exception as error:
                    return None, model_path, error

    return None, None, None

df = load_data()
xgb_model, xgb_model_path, model_load_error = load_model()

# Display summary statistics
st.header("Data Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", len(df))
col2.metric("Departments", df['EmpDepartment'].nunique())
col3.metric("Job Roles", df['EmpJobRole'].nunique())
col4.metric("Attrition Rate", f"{(df['Attrition'] == 'Yes').sum() / len(df) * 100:.1f}%")

# Data viewer
st.header("Employee Data Table")
st.dataframe(df, width='stretch')

# Filter and analysis section
st.header("Data Analysis")

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Attrition Analysis",
    "Department Analysis",
    "Age Distribution",
    "Salary Analysis",
    "Hiring Model"
])

with tab1:
    st.subheader("Attrition by Department")
    attrition_dept = df.groupby('EmpDepartment')['Attrition'].apply(lambda x: (x == 'Yes').sum()).reset_index()
    attrition_dept.columns = ['Department', 'Attrition Count']
    fig, ax = plt.subplots()
    ax.bar(attrition_dept['Department'], attrition_dept['Attrition Count'])
    ax.set_ylabel('Count')
    st.pyplot(fig)

with tab2:
    st.subheader("Employee Count by Department")
    dept_counts = df['EmpDepartment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(dept_counts.values, labels=dept_counts.index, autopct='%1.1f%%')
    st.pyplot(fig)

with tab3:
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['Age'], bins=20, edgecolor='black')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

with tab4:
    st.subheader("Monthly Hourly Rate by Department")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='EmpHourlyRate', by='EmpDepartment', ax=ax)
    ax.set_xlabel('Department')
    ax.set_ylabel('Hourly Rate')
    st.pyplot(fig)

with tab5:
    st.subheader("Hiring Model: Predict Performance Rating")
    if xgb_model is None:
        if model_load_error is not None and xgb_model_path is not None:
            st.error(
                f"A model file was found at `{xgb_model_path}` but it failed to load: {type(model_load_error).__name__}: {model_load_error}"
            )
        else:
            st.warning(
                "XGBoost model pipeline not found. Make sure `xgb_model_pipeline.pkl` or `xgb_model_pipeline.joblib` exists in the project root or the app directory."
            )
    else:
        st.markdown(
            "Select candidate attributes below and generate a predicted `PerformanceRating`."
        )

        candidate = {
            'Age': st.slider('Age', int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].median())),
            'DistanceFromHome': st.slider('Distance From Home', int(df['DistanceFromHome'].min()), int(df['DistanceFromHome'].max()), int(df['DistanceFromHome'].median())),
            'EmpEducationLevel': st.slider('Education Level', int(df['EmpEducationLevel'].min()), int(df['EmpEducationLevel'].max()), int(df['EmpEducationLevel'].median())),
            'EmpEnvironmentSatisfaction': st.slider('Environment Satisfaction', int(df['EmpEnvironmentSatisfaction'].min()), int(df['EmpEnvironmentSatisfaction'].max()), int(df['EmpEnvironmentSatisfaction'].median())),
            'EmpHourlyRate': st.number_input('Hourly Rate', int(df['EmpHourlyRate'].min()), int(df['EmpHourlyRate'].max()), int(df['EmpHourlyRate'].median())),
            'EmpJobInvolvement': st.slider('Job Involvement', int(df['EmpJobInvolvement'].min()), int(df['EmpJobInvolvement'].max()), int(df['EmpJobInvolvement'].median())),
            'EmpJobLevel': st.slider('Job Level', int(df['EmpJobLevel'].min()), int(df['EmpJobLevel'].max()), int(df['EmpJobLevel'].median())),
            'EmpJobSatisfaction': st.slider('Job Satisfaction', int(df['EmpJobSatisfaction'].min()), int(df['EmpJobSatisfaction'].max()), int(df['EmpJobSatisfaction'].median())),
            'NumCompaniesWorked': st.slider('Number of Companies Worked', int(df['NumCompaniesWorked'].min()), int(df['NumCompaniesWorked'].max()), int(df['NumCompaniesWorked'].median())),
            'EmpLastSalaryHikePercent': st.slider('Last Salary Hike (%)', int(df['EmpLastSalaryHikePercent'].min()), int(df['EmpLastSalaryHikePercent'].max()), int(df['EmpLastSalaryHikePercent'].median())),
            'EmpRelationshipSatisfaction': st.slider('Relationship Satisfaction', int(df['EmpRelationshipSatisfaction'].min()), int(df['EmpRelationshipSatisfaction'].max()), int(df['EmpRelationshipSatisfaction'].median())),
            'TotalWorkExperienceInYears': st.slider('Total Work Experience', int(df['TotalWorkExperienceInYears'].min()), int(df['TotalWorkExperienceInYears'].max()), int(df['TotalWorkExperienceInYears'].median())),
            'TrainingTimesLastYear': st.slider('Training Times Last Year', int(df['TrainingTimesLastYear'].min()), int(df['TrainingTimesLastYear'].max()), int(df['TrainingTimesLastYear'].median())),
            'EmpWorkLifeBalance': st.slider('Work-Life Balance', int(df['EmpWorkLifeBalance'].min()), int(df['EmpWorkLifeBalance'].max()), int(df['EmpWorkLifeBalance'].median())),
            'ExperienceYearsAtThisCompany': st.slider('Years at Company', int(df['ExperienceYearsAtThisCompany'].min()), int(df['ExperienceYearsAtThisCompany'].max()), int(df['ExperienceYearsAtThisCompany'].median())),
            'ExperienceYearsInCurrentRole': st.slider('Years in Current Role', int(df['ExperienceYearsInCurrentRole'].min()), int(df['ExperienceYearsInCurrentRole'].max()), int(df['ExperienceYearsInCurrentRole'].median())),
            'YearsSinceLastPromotion': st.slider('Years Since Last Promotion', int(df['YearsSinceLastPromotion'].min()), int(df['YearsSinceLastPromotion'].max()), int(df['YearsSinceLastPromotion'].median())),
            'YearsWithCurrManager': st.slider('Years with Current Manager', int(df['YearsWithCurrManager'].min()), int(df['YearsWithCurrManager'].max()), int(df['YearsWithCurrManager'].median())),
            'Gender': st.selectbox('Gender', sorted(df['Gender'].dropna().unique())),
            'EducationBackground': st.selectbox('Education Background', sorted(df['EducationBackground'].dropna().unique())),
            'MaritalStatus': st.selectbox('Marital Status', sorted(df['MaritalStatus'].dropna().unique())),
            'EmpDepartment': st.selectbox('Department', sorted(df['EmpDepartment'].dropna().unique())),
            'EmpJobRole': st.selectbox('Job Role', sorted(df['EmpJobRole'].dropna().unique())),
            'BusinessTravelFrequency': st.selectbox('Business Travel Frequency', sorted(df['BusinessTravelFrequency'].dropna().unique())),
            'OverTime': st.selectbox('OverTime', sorted(df['OverTime'].dropna().unique())),
        }

        candidate_df = pd.DataFrame([candidate])
        if st.button('Predict Performance Rating'):
            prediction = xgb_model.predict(candidate_df)[0]
            st.success(f"Predicted Performance Rating: {prediction}")
            if hasattr(xgb_model, 'predict_proba'):
                proba = xgb_model.predict_proba(candidate_df)[0]
                st.write('Prediction confidence:')
                st.write({f'Rating {cls}': f'{prob:.2%}' for cls, prob in enumerate(proba, start=1)})

# Filter section
st.header("Filter Data")
col1, col2, col3 = st.columns(3)

with col1:
    selected_dept = st.multiselect(
        "Select Department(s)",
        df['EmpDepartment'].unique(),
        default=df['EmpDepartment'].unique()
    )

with col2:
    selected_job = st.multiselect(
        "Select Job Role(s)",
        df['EmpJobRole'].unique(),
        default=df['EmpJobRole'].unique()
    )

with col3:
    attrition_filter = st.selectbox(
        "Attrition Status",
        ["All", "Yes", "No"]
    )

# Apply filters
filtered_df = df[
    (df['EmpDepartment'].isin(selected_dept)) &
    (df['EmpJobRole'].isin(selected_job))
]

if attrition_filter != "All":
    filtered_df = filtered_df[filtered_df['Attrition'] == attrition_filter]

st.subheader("Filtered Results")
st.dataframe(filtered_df, width='stretch')
st.write(f"Total records: {len(filtered_df)}")
