import streamlit as st
import kagglehub
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pycountry
import pycountry_convert as pc

def get_region_from_country(alpha3):
    try:
        alpha2 = pc.country_alpha3_to_country_alpha2(alpha3)
        continent_code = pc.country_alpha2_to_continent_code(alpha2)
        continent_dict = {
            'NA': 'North America',
            'SA': 'South America',
            'OC': 'Oceania',
            'AF': 'Africa',
            'EU': 'Europe',
            'AS': 'Asia'
        }
        region = continent_dict.get(continent_code, 'Other')
        if alpha3 == "JPN":
            region = "Japan"
        elif region not in ['North America', 'Europe']:
            region = "Rest of World"
        return region
    except Exception as e:
        return "Other"

path = kagglehub.dataset_download("anandshaw2001/video-game-sales")
df = pd.read_csv(path + "/vgsales.csv")

df = df[df["Year"] != "N/A"]
df = df[df["Year"].notna()]
df = df[df["Year"] != 0]
df = df[df["Publisher"] != 0]
df = df[df["Publisher"].notna()]
df["Platform"] = df["Platform"].astype(str)
df["Year"] = df["Year"].fillna(0).astype(int)
df = df[df["Global_Sales"].notna()]

st.set_page_config(
    page_title="Video Game Sales Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "Link to the dataset:  \n[Video Game Sales Dataset](https://www.kaggle.com/datasets/anandshaw2001/video-game-sales)"
    }
)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>🎮 Video Game Sales Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Filter Options")

with st.sidebar.expander("Search for a Game", expanded=False):
    game_search = st.text_input("Game Name")
    
year_filter = st.sidebar.slider(
    "Select Release Year", 
    int(df['Year'].min()), int(df['Year'].max()),
    (int(df['Year'].min()), int(df['Year'].max()))
)

def multi_select_all(label, options):
    options = sorted(list(options))
    new_options = ["All"] + options
    selected = st.multiselect(label, options=new_options, default=["All"])
    if "All" in selected:
        return options
    else:
        return selected

with st.sidebar.expander("Select Genre", expanded=False):
    genre_filter = multi_select_all("Genre", df['Genre'].unique())

with st.sidebar.expander("Select Platform", expanded=False):
    platform_filter = multi_select_all("Platform", df['Platform'].unique())

with st.sidebar.expander("Select Top Publishers", expanded=False):
    publisher_filter = multi_select_all("Publisher (All)", df['Publisher'].unique())

df_filtered = df[(df['Genre'].isin(genre_filter)) &
                 (df['Publisher'].isin(publisher_filter)) &
                 (df['Platform'].isin(platform_filter)) &
                 (df['Year'] >= year_filter[0]) &
                 (df['Year'] <= year_filter[1])]
if game_search:
    df_filtered = df_filtered[df_filtered['Name'].str.contains(game_search, case=False)]

tab_overview, tab_global_sales, tab_sales_by_region, tab_data_analytics = st.tabs(["Overview", "Global Sales", "Sales by Region", "Data Analytics"])

with tab_overview:
    st.subheader("Dashboard Overview")
    col1, col2 = st.columns((1, 2))
    with col1:
        top_game = df_filtered.sort_values(by="Global_Sales", ascending=False).iloc[0]
        st.metric("Top Game", top_game['Name'], f"{top_game['Global_Sales']}M sales")
        
        total_sales = df_filtered['Global_Sales'].sum()
        st.metric("Total Global Sales", f"{total_sales:.2f}M")
        
        top_genre = df_filtered.groupby('Genre')['Global_Sales'].sum().idxmax()
        st.metric("Top Genre", top_genre)
        
        total_games = df_filtered['Name'].nunique()
        st.metric("Total Games", total_games)
    with col2:
        st.subheader("Top 10 Games by Global Sales")
        top_10_games = df_filtered.sort_values(by="Global_Sales", ascending=False).head(10)
        fig_top10 = px.bar(top_10_games, x="Name", y="Global_Sales", 
                           title="Top 10 Games by Global Sales", color="Global_Sales")
        st.plotly_chart(fig_top10, use_container_width=True)
    
    st.markdown("### Data Table")
    st.dataframe(df_filtered)

with tab_global_sales:
    st.subheader("Global Sales Trend Over Years")
    sales_by_year = df_filtered.groupby("Year")["Global_Sales"].sum().reset_index()
    fig_line = px.line(sales_by_year, x="Year", y="Global_Sales", title="Global Sales Over Time")
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.subheader("Global Sales by Genre (Bar Chart)")
    genre_sales = df_filtered.groupby("Genre")["Global_Sales"].sum().reset_index()
    fig_bar = px.bar(genre_sales, x="Genre", y="Global_Sales", title="Global Sales by Genre", color="Genre")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.subheader("Pie Chart for Genre Distribution")
    fig_pie = px.pie(genre_sales, values='Global_Sales', names='Genre', title='Sales Distribution by Genre')
    st.plotly_chart(fig_pie)
    
    st.subheader("Global Sales by Platform (Bar Chart)")
    platform_sales = df_filtered.groupby("Platform")["Global_Sales"].sum().reset_index()
    fig_platform = px.bar(platform_sales, x="Platform", y="Global_Sales", title="Global Sales by Platform", color="Platform")
    fig_platform.update_xaxes(type='category')
    st.plotly_chart(fig_platform, use_container_width=True)

    st.subheader("Top 10 Publishers by Global Sales (Bar Chart)")
    top_publishers = df_filtered.groupby("Publisher")["Global_Sales"].sum().reset_index()
    top_publishers = top_publishers.sort_values(by="Global_Sales", ascending=False).head(10)
    fig_top_publishers = px.bar(top_publishers, x="Publisher", y="Global_Sales", title="Top 10 Publishers by Global Sales", color="Global_Sales")
    fig_top_publishers.update_xaxes(type='category')
    st.plotly_chart(fig_top_publishers, use_container_width=True)    

with tab_sales_by_region:
    st.subheader("Global Sales by Region (Bar Chart)")
    region_sales = df_filtered.groupby("Platform")[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().reset_index()
    fig_region = px.bar(region_sales, x="Platform", y=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
                        title="Sales by Region", barmode='group')
    st.plotly_chart(fig_region, use_container_width=True)
    
    st.subheader("Choropleth Map of Global Sales by Region")
    na_sales = df_filtered['NA_Sales'].sum()
    eu_sales = df_filtered['EU_Sales'].sum()
    jp_sales = df_filtered['JP_Sales'].sum()
    other_sales = df_filtered['Other_Sales'].sum()
    
    rows = []
    for country in list(pycountry.countries):
        alpha3 = country.alpha_3
        region = get_region_from_country(alpha3)
        if region == "North America":
            sales = na_sales
        elif region == "Europe":
            sales = eu_sales
        elif region == "Japan":
            sales = jp_sales
        else:
            sales = other_sales
        rows.append({"Country": alpha3, "Sales": sales, "Region": region})
    
    df_choropleth = pd.DataFrame(rows)
    
    fig_choropleth = px.choropleth(
        df_choropleth,
        locations="Country",
        locationmode="ISO-3",
        color="Sales",
        hover_name="Region",
        labels={"Sales": "Sales (in millions)"},
        title="Global Sales by Region (Based on Official ISO Country Codes)"
    )
    fig_choropleth.update_geos(
        resolution=50,
        showcoastlines=True,
        coastlinecolor="white",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="blue",
        showcountries=True,
        countrycolor="gray",
        countrywidth=2,
        projection_type="natural earth"
    )
    fig_choropleth.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title_x=0.5
    )
    st.plotly_chart(fig_choropleth, use_container_width=True)

with tab_data_analytics:
    st.header("Data Analytics")
    st.write("Here are some additional insights based on the filtered data:")
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df_filtered.describe())
    
    st.subheader("Correlation Heatmap (Altair)")
    corr_df_alt = df_filtered[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].corr().reset_index().melt(id_vars='index')
    heatmap_alt = alt.Chart(corr_df_alt).mark_rect().encode(
        x=alt.X('index:N', title=""),
        y=alt.Y('variable:N', title=""),
        color=alt.Color('value:Q'),
        tooltip=['index', 'variable', 'value']
    ).properties(
        width=300,
        height=300,
        title="Sales Correlation Heatmap (Altair)"
    )
    st.altair_chart(heatmap_alt, use_container_width=True)

    #st.subheader("Correlation Heatmap (Seaborn)")
    #corr_df = df_filtered[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].corr()
    #fig_heatmap = plt.figure(figsize=(10, 6))
    #sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    #plt.title("Correlation Heatmap")
    #st.pyplot(fig_heatmap)