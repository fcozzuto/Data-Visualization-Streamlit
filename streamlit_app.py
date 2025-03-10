import streamlit as st
import kagglehub
import os
import pandas as pd
import altair as alt

st.title("Streamlit Python Demo Dashboard")
st.write("This is a simple Dashboard created in Python, using the streamlit module and its functionalities.")

path = kagglehub.dataset_download("ahmedmohamed2003/quality-of-life-for-each-country")
print(os.listdir(path))

df = pd.read_csv(path + "/Quality_of_Life.csv")

# Setting up the parameters to display the chart 
chart = alt.Chart(df).mark_line().encode(
    x=alt.X("country", title="Country"),
    y=alt.Y("Safety Value", title="Safety Value")
).properties(
    title="Safety Value by Country"
)

# Displaying the chart in Streamlit
st.altair_chart(chart, use_container_width=True)