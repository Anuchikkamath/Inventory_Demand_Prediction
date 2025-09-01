import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Custom CSS for Background Image
background_image_url = "https://www.veeqo.com/_next/image?url=https%3A%2F%2Fimages.ctfassets.net%2Fhfb264dqso7g%2F4QovdfnQYcBgP0X4PN0QfM%2F6313238ef1c4def0614eb43b209bbf5b%2Fdemand-forecasting.jpg&w=2560&q=75"  # Replace with your image URL

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("{background_image_url}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stSidebar"] {{
    background-color: rgba(0, 0, 0, 0.7);  /* Optional: Darken sidebar for contrast */
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

with st.sidebar.expander("⚙️ How It Works"):
    st.markdown("""
    1️⃣ **Upload your inventory data.**  
    2️⃣ **Adjust the input parameters if needed.**  
    3️⃣ **Click the 'Forecast Demand' button.**  
    4️⃣ **Get demand forecasts for your product and actionable restocking suggestions!**  
    """)

st.title("🌟Inventory Demand Forecast🌟")

import streamlit as st
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\hp\Downloads\archive (11)\retail_store_inventory.csv")
df_sampled = df.sample(n=20000, random_state=42)


import streamlit as st

import streamlit as st

# Custom CSS
st.markdown(
    """
    <style>
        /* Middle section styling */
        .middle-section {
            color: white !important;
        }

        /* Forecast button styling */
        .forecast-button {
            background-color: black !important;
            color: white !important;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
        }

        /* Sidebar headings */
        .sidebar-heading {
            color: white !important;
            font-size: 18px;
            font-weight: bold;
        }

        /* Dropdown items styling */
        .dropdown select {
            background-color: white !important;
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("""
    ## 🏢 How This App Helps Your Business?

    - 📊 **Real-time Demand Analysis** – Understand which products are in demand.  
    - 📦 **Efficient Inventory Management** – Restock before running out!  
    - 📅 **Seasonal Trend Insights** – Plan ahead for peak seasons and promotions.  
    - 📉 **Cost Reduction** – Reduce unnecessary purchases of slow-moving products.  
    - 🚚 **Better Supplier Negotiation** – Order in bulk for cost benefits.  
""")



st.write("""
    ## 💰 How Accurate Demand Forecasting Saves Money?

    **Before Forecasting:**  
    - ❌ Overstocking leads to **30% inventory waste** 💸  
    - ❌ **Stockouts = Lost Sales** (~20% revenue loss)  
    - ❌ **Emergency Orders** increase costs 🚚  

    **After Using This App:**  
    - ✅ **Optimal Stock Levels** – No more waste!  
    - ✅ **10-20% Higher Revenue** – Always meet customer demand.  
    - ✅ **Smarter Reordering** – No need for costly rush orders.  
""")




df_sampled['Date'] = pd.to_datetime(df_sampled['Date'])
df_sampled['Day'] = df_sampled['Date'].dt.day
df_sampled['Month_Name'] = df_sampled['Date'].dt.month_name()
df_sampled['Year'] = df_sampled['Date'].dt.year

selected_graph = st.sidebar.selectbox(
    "📊 Select a Visualization:", 
    ["None", "📈 Demand Trend Over Time", "📦 Inventory Levels Over Time", "🏆 Top 10 Products"]
)

def plot_demand_trend():
        st.markdown("<h3 style='color:white;'>📈Demand Trend Over Time</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(3, 2), facecolor = 'black') # Adjust width and heigh
        df_sampled.groupby('Month_Name')['Units Ordered'].sum().plot(ax=ax, color='cyan')
        ax.set_title("Demand Trend Over Time", color='white', fontsize=14)
        ax.set_xlabel("Months", color='white', fontsize=12)
        ax.set_ylabel("Units Ordered", color='white', fontsize=12)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig)

def plot_inventory_levels():
    st.markdown("<h3 style='color:white;'>📊Inventory Levels Over Time</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(3, 2), facecolor = 'black')
    df_sampled.groupby('Month_Name')['Inventory Level'].sum().plot(ax=ax, color='red')
    ax.set_title("Inventory Levels Over Time", color='white', fontsize=14)
    ax.set_xlabel("Months", color='white', fontsize=12)
    ax.set_ylabel("Inventory Level", color='white', fontsize=12)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    st.pyplot(fig)

def plot_top_10_products():
    st.markdown("<h3 style='color:white;'>🏆Top 10 Selling Products</h3>", unsafe_allow_html=True)
    top_products = df_sampled.groupby("Product ID")["Units Ordered"].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(3, 2), facecolor = 'black')
    top_products.plot(kind="bar", ax=ax, color="orange")
    ax.set_title("Top 10 Selling Products", color = 'white', fontsize = 14)
    ax.set_ylabel("Total Units Ordered", color = 'white', fontsize = 12)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    st.pyplot(fig)

# Display selected graph dynamically
if selected_graph == "📈 Demand Trend Over Time":
    plot_demand_trend()
elif selected_graph == "📦 Inventory Levels Over Time":
    plot_inventory_levels()
elif selected_graph == "🏆 Top 10 Products":
    plot_top_10_products()



df_sampled.drop(columns = ['Date'], axis = 1, inplace = True)
df_sampled.drop(columns = ['Region', 'Weather Condition', 'Holiday/Promotion', 'Seasonality'], 
                axis = 1, inplace = True)
df_sampled.drop(columns = ['Discount', 'Competitor Pricing'], axis = 1, inplace = True)

X = df_sampled.drop(columns = ['Demand Forecast'], axis = 1)
y = df_sampled['Demand Forecast']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

Cont_col = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price']
Ordinal_col = ['Store ID', 'Product ID', 'Day','Month_Name','Year']
Nominal_col = ['Category']

orders = [['S001', 'S002', 'S003', 'S004', 'S005'],
          ['P0001', 'P0002', 'P0003', 'P0004', 'P0005', 'P0006', 'P0007', 'P0008', 'P0009',
          'P0010', 'P0011', 'P0012', 'P0013', 'P0014', 'P0015', 'P0016', 'P0017', 'P0018', 'P0019', 'P0020'],
          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
          ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
          ['2022', '2023', '2024']]

cont_transform = Pipeline([("Scaling", MinMaxScaler()),
                      ("Transform", PowerTransformer(method = 'yeo-johnson'))])

nominal_transform = Pipeline([("Encoding1", OneHotEncoder())])

ordinal_transform = Pipeline([("Encoding2", OrdinalEncoder(categories = orders))])

preprocess = ColumnTransformer([("Pipeline_1", cont_transform, Cont_col),
                                ("Pipeline_2", nominal_transform, Nominal_col),
                                ("Pipeline_3", ordinal_transform, Ordinal_col)],
                              remainder = 'passthrough')

pipeline = Pipeline([("Feature_Engineer", preprocess),
                  ("model", RandomForestRegressor())])


pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
error = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# User Prediction

st.sidebar.header("Enter your Inventory details here")
User_Store_ID = st.sidebar.selectbox("Store ID", ('S001', 'S002', 'S003', 'S004', 'S005'))
User_Product_ID = st.sidebar.selectbox("Product ID", ('P0001', 'P0002', 'P0003', 'P0004', 'P0005', 'P0006', 
                            'P0007', 'P0008', 'P0009','P0010', 'P0011', 'P0012', 'P0013', 'P0014', 
                            'P0015', 'P0016', 'P0017', 'P0018', 'P0019', 'P0020'))
User_Category = st.sidebar.selectbox("Category", ('Groceries', 'Toys', 'Electronics', 'Furniture', 'Clothing'))
User_Inventory_level = st.sidebar.number_input("Inventory Level")
User_Unit_sold = st.sidebar.number_input("Units Sold")
User_Units_ordered = st.sidebar.number_input("Units Ordered")
User_Price = st.sidebar.number_input("Price")
User_Day = st.sidebar.selectbox("Day", range(1, 32))
User_month = st.sidebar.selectbox("Month_Name", ('January', 'February', 'March', 'April', 'May', 'June', 'July', 
                                         'August', 'September', 'October', 'November', 'December'))
User_year = st.sidebar.selectbox("Year", (2022,2023,2024))

User_input_data = pd.DataFrame([[User_Store_ID, User_Product_ID, User_Category, 
                                 User_Inventory_level, User_Unit_sold, User_Units_ordered, User_Price, 
                                 User_Day, User_month, User_year]],
                          columns=['Store ID', 'Product ID', 'Category', 
                                   'Inventory Level', 'Units Sold', 'Units Ordered', 'Price',
                                   'Day', 'Month_Name', 'Year'])

# Suggesting Restocking level for particular product

if st.button("Forecast Demand", key="forecast_demand_button"):
    demand_forecast = pipeline.predict(User_input_data)
    st.write(demand_forecast)
    # Suggesting Restocking level for particular product
    safety_stock = 10  
    if User_Inventory_level < (demand_forecast + User_Units_ordered + safety_stock):
           restocking_level = (demand_forecast + User_Units_ordered + safety_stock) - User_Inventory_level
           st.error(f"Your inventory level is lower than the forecasted demand!! Please consider restocking your product.", icon="🚨")
           st.info(f"Recommended Restocking: Restock {restocking_level} units for Product {User_Product_ID}.", icon="✅")
    else:
        st.success(f"✅You are good to go. No restocking needed for Product {User_Product_ID}")

