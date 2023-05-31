import streamlit as st
import pandas as pd

st.set_page_config(page_title="HDB Resake Prices", page_icon=":house:", layout="wide", initial_sidebar_state="auto", menu_items={
        'Report a bug': "limsienlong@gmail.com",
        'About': 'https://www.natuyuki.pythonanywhere.com'
    })

# Custom functions and decorators
@st.cache_data
def load_data(url):
    df = pd.read_csv(url, index_col=0)
    return df

# Side bar stuff
st.sidebar.write("Select the scope of the data to begin")
year_select = st.sidebar.selectbox(
    'Year',
    (2023, 2022)
)

df = load_data(f'static/{year_select}_01_to_04.csv')

month_select = st.sidebar.multiselect(
    'Month', ['All']+list(df['month'].sort_values(ascending=True).unique())
)

# Main content
st.title('HDB Resale Housing Price:')
st.title('Dataset Exploration')
st.write("Use the sidebar and sliders available to explore the Dataset")

if 'All' not in month_select:
    df = df[df['month'].isin(month_select)]

# Dataframe section
(left_column, right_column) = st.columns(2)
rows = left_column.slider("Rows to show", min_value=100, max_value=len(df), step=1000)

display_df = right_column.selectbox('', ('Show Dataframe', 'Hide Dataframe'))
if display_df=='Show Dataframe':
    st.write(df.head(rows))
st.divider()

# Plot section
st.subheader("Graphing")
plot_type = st.selectbox('Select plot type', ['Bar chart', 'Line graph'])
(left_column_plot, right_column_plot) = st.columns(2)
y_axis = right_column_plot.selectbox('Select y-axis', df.columns)
aggregate = left_column_plot.selectbox('Aggregate', ['None', 'Mean', 'Median', 'Count','Sum'])
if aggregate != 'None':
    groupby = left_column_plot.selectbox('Group by', df.columns)
    # Actions to aggregate
    if aggregate=='Sum':
        gb_df = df.groupby(groupby).sum(numeric_only=True)
    elif aggregate=='Mean':
        gb_df = df.groupby(groupby).mean(numeric_only=True)
    elif aggregate=='Count':
        gb_df = df.groupby(groupby).count(numeric_only=True)
    elif aggregate=='Median':
        gb_df = df.groupby(groupby).median(numeric_only=True)
    x_axis=None
else:
    x_axis = left_column_plot.selectbox('Select x-axis', df.columns)

st.subheader(f'{plot_type}')
if plot_type=='Bar chart':
    if x_axis:
        st.bar_chart(data=df, x=x_axis, y=y_axis)
    else:
        st.bar_chart(data=gb_df, y=y_axis)
elif plot_type=='Line graph':
    if x_axis:
        st.line_chart(data=df, x=x_axis, y=y_axis)
    else:
        st.line_chart(data=gb_df, y=y_axis)

st.divider()

# Map
st.subheader('Map of data')
map_data = df.loc[:,['latitude', 'longitude']]
st.map(data=map_data, zoom=11)