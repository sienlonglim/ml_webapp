import streamlit as st
import pandas as pd

st.set_page_config(page_title="HDB Resale Prices", 
                   page_icon=":house:", 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items={'About': "Welcome to Sien Long's Dataset Explorer for HDB Resale Prices"})

# Custom functions and decorators
@st.cache_data
def load_data(url):
    df = pd.read_csv(url, index_col=0)
    return df

# Year scope to load df file
st.sidebar.write("Select the scope of the data to begin")
year_select = st.sidebar.selectbox(
    'Year',
    (2023, 2022)
)

df = load_data(f'static/{year_select}.csv')

# Month and Price slicer
month_select = st.sidebar.multiselect('Month', 
                                      ['All']+list(df['month'].sort_values(ascending=True).unique()),
                                      default='All'
)

towns = st.sidebar.multiselect('Towns', 
                                ['All']+list(df['town'].sort_values(ascending=True).unique()),
                                default='All'
)

price_range = st.sidebar.slider("Price range (SGD)", 
                                value=(0, int(df['resale_price'].max())), # Makes this into a range slide
                                step=50000)

# Filter df based on slicers
if 'All' not in month_select:
    df = df[df['month'].isin(month_select)]

if 'All' not in towns:
    df = df[df['town'].isin(towns)]

lower_range = df['resale_price']>=price_range[0]
upper_range = df['resale_price']<=price_range[1]
df = df[lower_range & upper_range]


# Main content
st.title('HDB Resale Housing Price:')
st.title('Dataset Exploration')
st.write("Use the sidebar and sliders available to explore the Dataset")


# Dataframe section
(left_column, right_column) = st.columns(2)
rows = left_column.slider("Rows to show", min_value=100, max_value=len(df), step=1000)

display_df = right_column.selectbox('Dataframe Visibility', ('Show Dataframe', 'Hide Dataframe'), label_visibility='hidden')
if display_df=='Show Dataframe':
    st.write(df.head(rows))

st.write('---')

# Plot section
st.subheader("Data Visualization")
plot_type = st.selectbox('Select plot type', ['Bar chart', 'Line graph'])
aggregate = st.checkbox("Aggregate", key="enable_aggregate", 
                        help='Aggregates by the Aggregate measure and Group by (x-axis)',
                        value=True)

# Create two columns for plotting options
(left_column_plot, right_column_plot) = st.columns(2)

# Disables aggregate by default
aggregate_measure = left_column_plot.selectbox('Aggregate measure', ['Mean', 'Median', 'Count','Sum'], 
                                               disabled=not st.session_state.enable_aggregate)
groupby = left_column_plot.selectbox('Group by (x-axis)', df.columns, index=5, disabled=not st.session_state.enable_aggregate)

# Disable x_axis is aggregate is selected
x_axis = right_column_plot.selectbox('Select x-axis', df.columns, index=5 ,disabled=st.session_state.enable_aggregate)
y_axis = right_column_plot.selectbox('Select y-axis', df.columns)

if st.session_state.enable_aggregate:
    # Actions to aggregate df
    if aggregate_measure=='Sum':
        gb_df = df.groupby(groupby).sum(numeric_only=True)
    elif aggregate_measure=='Mean':
        gb_df = df.groupby(groupby).mean(numeric_only=True)
    elif aggregate_measure=='Count':
        gb_df = df.groupby(groupby).count()
    elif aggregate_measure=='Median':
        gb_df = df.groupby(groupby).median(numeric_only=True)
    
# Final chart plot decision
st.subheader(f'{plot_type}')
if plot_type=='Bar chart':
    if st.session_state.enable_aggregate:
        st.bar_chart(data=gb_df, y=y_axis)
    else:
        st.bar_chart(data=df, x=x_axis, y=y_axis)
if plot_type=='Line graph':
    if st.session_state.enable_aggregate:
        st.line_chart(data=gb_df, y=y_axis)
    else:
        st.line_chart(data=df, x=x_axis, y=y_axis)

st.write('---')

# Map
st.subheader('Map of data')
map_data = df.loc[:,['latitude', 'longitude']]
st.map(data=map_data, zoom=11)