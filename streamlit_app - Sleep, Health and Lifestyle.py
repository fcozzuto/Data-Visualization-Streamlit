import streamlit as st
import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")
print(os.listdir(path))

df = pd.read_csv(path + "/Sleep_health_and_lifestyle_dataset.csv")
    
df.fillna("Unknown", inplace=True)
df.drop("Person ID", axis=1, inplace=True)  

st.set_page_config(
    page_title="Sleep, Health & Lifestyle Dashboard",
    page_icon="🌜",
    layout="wide",
    #layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "Link to the dataset\:  \n[Sleep, Health & Lifestyle Dataset](https://www.kaggle.com/uom190346a/sleep-health-and-lifestyle-dataset)"
    }
)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>🌜Sleep, Health & Lifestyle Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)


#st.title("Sleep, Health & Lifestyle Dashboard")
#st.write("This is a simple Dashboard created in Python, using the streamlit module and its functionalities.")

st.sidebar.header("Filter Options")

gender_options = sorted(df['Gender'].unique().tolist())
selected_gender = st.sidebar.selectbox("Select Gender", ["All"] + gender_options)
if selected_gender != "All":
    df = df[df["Gender"] == selected_gender]

occupation_options = sorted(df['Occupation'].unique().tolist())
selected_occupation = st.sidebar.selectbox("Select Occupation", ["All"] + occupation_options)
if selected_occupation != "All":
    df = df[df["Occupation"] == selected_occupation]

quality_options = sorted(df['Quality of Sleep'].unique().tolist())
selected_quality = st.sidebar.multiselect("Select Quality of Sleep", quality_options, default=quality_options)
df = df[df['Quality of Sleep'].isin(selected_quality)]

min_age = int(df['Age'].min())
max_age = int(df['Age'].max())
age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))
df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]

clustering_df = df[['Quality of Sleep', 'Stress Level']].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_df)

kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
clustering_df['Cluster'] = clusters


#col1, col2, col3 = st.columns((3, 3, 2), gap='small')#
#with col1:
#    st.subheader("Correlation Heatmap (Numeric Vars)")
#    numeric_df = df.select_dtypes(include=['float64', 'int64'])
#    if not numeric_df.empty:
#        fig2, ax2 = plt.subplots(figsize=(12, 8))
#        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
#        ax2.set_title("Correlation Matrix")
#        st.pyplot(fig2)
#
#    st.subheader("Age vs. Sleep Duration Regression")
#    fig_reg, ax_reg = plt.subplots(figsize=(10,6))
#    sns.regplot(x="Age", y="Sleep Duration", data=df, ax=ax_reg, scatter_kws={'alpha':0.6})
#    ax_reg.set_title("Age vs. Sleep Duration")
#    st.pyplot(fig_reg)
#
#with col2:
#    st.subheader("Distribution of Sleep Duration")
#    fig, ax = plt.subplots(figsize=(10, 6))
#    #sns.histplot(df['Sleep Duration'], kde=True, color="skyblue", ax=ax)
#    hist = alt.Chart(df).mark_bar().encode(
#        alt.X("Sleep Duration", bin=alt.Bin(maxbins=30), title="Hours of Sleep"),
#        alt.Y('count()', title="Number of People"),
#        tooltip=['count()']
#    )
#    st.altair_chart(hist, use_container_width=True)
#
#    st.subheader("K-Means Clustering on Stress Level and Quality of Sleep")
#    fig3, ax3 = plt.subplots(figsize=(10, 6))
#    sns.scatterplot(x=clustering_df['Stress Level'], y=clustering_df['Quality of Sleep'], hue=clustering_df['Cluster'], palette="viridis")
#    st.pyplot(fig3)
#
#with col3:
#    st.subheader("Dataset Preview")
#    st.write(df.head().T)
#    st.write(df.describe().T)


tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "Advanced Analytics"])

with tab1:
    st.subheader("Overview")
    st.write(df.head())
    st.write(df.describe())

with tab2:
    st.subheader("Distribution of Sleep Duration")
    fig, ax = plt.subplots(figsize=(10, 6))
    #sns.histplot(df['Sleep Duration'], kde=True, color="skyblue", ax=ax)
    hist = alt.Chart(df).mark_bar().encode(
        alt.X("Sleep Duration", bin=alt.Bin(maxbins=30), title="Hours of Sleep"),
        alt.Y('count()', title="Number of People"),
        tooltip=['count()']
    )
    st.altair_chart(hist, use_container_width=True)

    st.subheader("K-Means Clustering on Stress Level and Quality of Sleep")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=clustering_df['Stress Level'], y=clustering_df['Quality of Sleep'], hue=clustering_df['Cluster'], palette="viridis")
    st.pyplot(fig3)

with tab3:
    st.subheader("Correlation Heatmap (Numeric Vars)")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        ax2.set_title("Correlation Matrix")
        st.pyplot(fig2)

    st.subheader("Age vs. Sleep Duration Regression")
    fig_reg, ax_reg = plt.subplots(figsize=(10,6))
    sns.regplot(x="Age", y="Sleep Duration", data=df, ax=ax_reg, scatter_kws={'alpha':0.6})
    ax_reg.set_title("Age vs. Sleep Duration")
    st.pyplot(fig_reg)#