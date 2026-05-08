import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="INX Future Inc. Employee Data Dashboard", layout="wide")

# Title
st.title("INX Future Inc. Employee Data Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('inx_data.csv')
    return df

df = load_data()

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
tab1, tab2, tab3, tab4 = st.tabs(["Attrition Analysis", "Department Analysis", "Age Distribution", "Salary Analysis"])

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
