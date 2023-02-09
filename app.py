import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pickle

def model(csv):
    # Load data
    df = pd.read_csv(csv, encoding="latin-1")
    scaler = StandardScaler()
    XX = scaler.fit_transform(df)
    X = XX[:, 1:-1]
    y = XX[:, -1]
    df = pd.DataFrame(XX, columns=df.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # Load model
    model = pickle.load(open("./Laptop_price_predictions/tree.sav", "rb"))
    
    # evaluation
    y_pred = model.predict(X_test)
    training_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 2
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return df, model, training_score, test_score, mae, mse, rmse, mape

(
    df, model, training_score,
    test_score, mae, mse, rmse,
    mape
) = model("./Laptop_price_predictions/laptop.csv")
df2 = pd.read_csv("./Laptop_price_predictions/laptop_price.csv", encoding="latin-1")

df.drop(df.columns[0], axis=1, inplace=True)
option = st.sidebar.selectbox(
    "Silakan pilih:",
    ("Home","Dataframe", "Data Visualization", "Model Building", "Predict")
)

if option == "Home" or option == "":
    st.write("""# Home Page""") #menampilkan halaman utama
    st.write()
    st.markdown("**This is my second app in streamlit**")
    st.write("This website is about build project about laptop predictions")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://dhonihanif.netlify.app/doni.jpeg", width=200)
    with col2:
        st.write(f"""
        Name : Dhoni Hanif Supriyadi\n
        Birth : 27 November 2001\n
        Degree : Bachelor degree start from 2020 until 2024\n
        Lasted GPA : 3.97 from 4.00\n
        University : University of Bina Sarana Informatika\n
        Field : Information System\n
        Linkedin : http://bit.ly/3x72z9F \n
        Github : https://github.com/dhonihanif \n
        Email : dhonihanif354@gmail.com \n
        Phone : +62895326168335
        """)

elif option == "Dataframe":
    st.write("""## Dataframe""") #menampilkan judul halaman dataframe
    st.write()
    st.markdown("**We read the data and do step of preparation data**")
    st.write(f"\nOriginal data with {df2.shape[0]} row and {df2.shape[1]} columns")
    st.write(df2)
    st.write(f"\nAfter cleaning and normalize with the data {df.shape[0]} row and {df.shape[1]} columns")
    st.write(df)

elif option == "Data Visualization":
    st.write("""## Data Visualization""")
    st.write()
    
    with st.expander("Graph of the number of laptops based on Type Name"):
        st.image("./Laptop_price_predictions/image/no1.png")
        st.write("""
        We can say that so much users use Notebook in this data and just little users who use Netbook. 
        People who use Gaming type is the second highest data of data TypeName.
        """)
    
    with st.expander("Graph of the minimum number of laptops based on Type Name"):
        st.image("./Laptop_price_predictions/image/no2.png")
        st.write("""
        We can say that:
- The minimum price, screen size, and weight of Workstation is higher than the other
- The minimum price of Netbook is lower than the other but the minimum screen size of 
Netbook is higher than 2 in 1 Convertible and the minimum weight of Netbook is higher than 
Ultrabook and 2 in 1 Convertible
- 2 in Convertible is the type that has higher minimum price than Notebook and Netbook but this 
type is the lowest minimum screen size and weight
        """)
    
    with st.expander("Graph of the maximum number of laptops based on Type Name"):
        st.image("./Laptop_price_predictions/image/no3.png")
        st.write("""
        From this, we can say that :
- The maximum of price, screen size, and weight is Gaming type
- Workstation is the highest minimum price, screen size, and weight than the other but has lower 
maximum price, screen size, and weight than Gaming
- Netbook is the lowest maximum price and screen size but this type has higher maximum weight 
than Ultrabook
- Ultrabook has higher maximum price than 2 in 1 Convertible and Netbook, but this type has 
lower screen size than 2 in 1 Convertible and higher than Netbook and this type is the lowest 
weight
        """)

    with st.expander("Graph of the mean of laptops based on Type Name"):
        st.image("./Laptop_price_predictions/image/no4.png")
        st.write("""
        From this, we can say that :
- The maximum of price, screen size, and weight is Gaming type
- Workstation is the highest minimum price, screen size, and weight than the other but has lower 
maximum price, screen size, and weight than Gaming
- Netbook is the lowest maximum price and screen size but this type has higher maximum weight 
than Ultrabook
- Ultrabook has higher maximum price than 2 in 1 Convertible and Netbook, but this type has 
lower screen size than 2 in 1 Convertible and higher than Netbook and this type is the lowest 
weight
        """)

    with st.expander("Graph of the standard deviation of laptops based on Type Name"):
        st.image("./Laptop_price_predictions/image/no5.png")
        st.write("""
        From this, we can say that:
- Gaming is the highest standard deviation in price euros and weight
- Netbook is the lowest standard deviation in price euros and screen size but has higher standard 
deviation in weight than Ultrabook
- 2 in 1 Convertible is the highest standard deviation in screen size
- Workstation is the second highest standard deviation in price euros and weight but has lower 
standard deviation in screen size than other except Netbook
        """)
    
    with st.expander("Graph of the correlation data"):
        st.image("./Laptop_price_predictions/image/no6.png")
        st.write("""
        From this, we can say that:
- Laptop_ID is has weak correlation with other data
- Inches has weak correlation with price but has high correlation with weight
- Weight has weak correlation with price but still higher than inches and has high correlation 
with inches
        """)
    
    with st.expander("Graph of the Laptop ID"):
        st.image("./Laptop_price_predictions/image/no7.png")
        st.write("""
        Like graph above, We know that Laptop_ID has high standard deviation but has weak correlation with other data.
        From this, we can say that:
        - This data have no unique values. Thats why this data has weak correlation with other data
        - We dont need this data. So, we can drop this data

        """)
    
    with st.expander("Graph of the correlation data with pairplot"):
        st.image("./Laptop_price_predictions/image/no8.png")
        st.write("""
        From this, we can say that:
- This data have no unique values. Thats why this data has weak correlation with other data
- We dont need this data. So, we can drop this data
        """)
    
    with st.expander("Univariate Analysis of Company"):
        st.image("./Laptop_price_predictions/image/no9.png")
        st.write("""
        From this, we can say that:
- The company that has highest values is Dell and Lenovo
- The company that has lowest values is LG, Fujitsu, Google, Huawei, and Chuwi
        """)
    
    with st.expander("Univariate Analysis of Type Name of laptop"):
        st.image("./Laptop_price_predictions/image/no10.png")
        st.write("""
        From this, we can say:
- The data Type which is highest value is Notebook. It means, so much people buy Notebook 
than other
- The data Type which is lowest value is Netbook. It just 1.9 % from all data TypeName
- Gaming is the second highest value after Notebook

        """)
    
    with st.expander("Univariate Analysis of Ram"):
        st.image("./Laptop_price_predictions/image/no11.png")
        st.write("""
        From this, we can say that:
- The data Ram which is highest value is 8GB. It's mean so much people use ram 8GB than other 
ram
- The data Ram which is lowest value is 64GB. It's 0.1 % people from all data that use 64GB
        """)
    
    with st.expander("Univariate Analysis of Memory"):
        st.image("./Laptop_price_predictions/image/no12.png")
        st.write("""
        From this, we can say that:
- The data Memory which is highest value is 256GB SSD. It's mean 31.6% people from all data 
use 256GB SSD
- The data Memory which is lowest value is very much that has just 0.1 % data
        """)
    with st.expander("Univariate Analysis of Operating System"):
        st.image("./Laptop_price_predictions/image/no13.png")
        st.write("""
        From this, we can say that:
- The data Operation System which is highest value is Windows 10. It's mean 82.3% people from 
all data use Windows 10
- The data Operation System which is lowest value is Android. It just 0.2 % people from all data 
use Android
        """)
    
    with st.expander("Univariate Analysis of Inches"):
        st.image("./Laptop_price_predictions/image/no14.png")
        st.write("""
        From this, we can say that:
- The highest data from inches data is around 15 - 16
- Data inches has 3 low outlier and 1 high outlier
- The minimum value is 10.1 and the maximum is 18.4
- The interquartile range (IQR) is between 14 and around 15
        """)
    
    with st.expander("Univariate Analysis of Weight"):
        st.image("./Laptop_price_predictions/image/no15.png")
        st.write("""
        From this, we can say that:
- The minimum value is 0.69 and the maximum value is 4.7
- Data has many high outlier
- The interquartile range (IQR) is between 1.5 and around 2

        """)
    
    with st.expander("Univariate Analysis of Price in Euros"):
        st.image("./Laptop_price_predictions/image/no16.png")
        st.write("""
        From this, we can say that:
- The minimum value is 0.69 and the maximum value is 4.7
- Data has many high outlier
- The interquartile range (IQR) is between 1.5 and around 2

        """)
    
    with st.expander("Bivariate Analysis of numerical variables Weight vs Inches"):
        st.image("./Laptop_price_predictions/image/no17.png")
        st.write("""
        From this, we can say that:
- These 2 data has positive correlation 
- The bigger weight has higher inches
- Some high weight has decreased
        """)
    
    with st.expander("Bivariate Analysis of numerical variables Weight vs Price in Euros"):
        st.image("./Laptop_price_predictions/image/no18.png")
        st.write("""
        From this, we can say that:
- Lots of data weight around 1.0 kg - 3.0 kg has low price
- some data has higher price with weight around 3.4kg - 3.5kg
- Data around 3kg has decreased greatly
- The maximum weight has high price which is above 3000 euro

        """)
    
    with st.expander("Bivariate Analysis of numerical variables Inches vs Prices in Euros"):
        st.image("./Laptop_price_predictions/image/no19.png")
        st.write("""
        From this, we can say that:
- The minimum of screen size has low price and the maximum of screen size has highest price
- Some data of screen size has low price and get decreased
- lots of data of screen size has high screen size and high price
        """)
    
    with st.expander("Bivariate Analysis for categorical variables Operating System vs Type Name"):
        st.image("./Laptop_price_predictions/image/no20.png")
        st.write("""
        From this, we can say that:
- Lots of people use Windows 10
- The highest type that people use in Windows 10 is Notebook and the lowest type that people 
use in Windows 10 is Netbook
- Android has just 1 type that people use and it is Workstation
        """)
    
    with st.expander("Bivariate Analysis for categorical variables Ram vs Type Name"):
        st.image("./Laptop_price_predictions/image/no21.png")
        st.write("""
        From This, we can say that:
- Lots of people use Ram 8GB and the highest is Notebook
- The highest value is Notebook
- Lots of people that use Ram 4GB use Notebook
- People who use 64GB use Gaming type
- Lots of people who use above 16GB use Gaming type
- Lots of people who use under 16GB use Notebook
        """)
    
    with st.expander("Bivariate Analysis for categorical variables Operating System vs Ram"):
        st.image("./Laptop_price_predictions/image/no22.png")
        st.write("""
        From this, we can say that:
- Lots of people who use macOS, No OS, Windows 10, and Windows 7 prefer use 8GB than 
other
- Most people using Windows 10 with Ram 8GB
- The highest people who use Linux is preper use Ram 4GB than 8GB
- 1 people use Android with Ram 4GB

        """)
        
    with st.expander("Bivariate Analysis for categorical variables Company vs Operating System"):
        st.image("./Laptop_price_predictions/image/no23.png")
        st.write("""
        From this, we can say that:
- Many companies sell Operation System Windows 10 than other Operation System
- Dell has the highest sell for Windows 10
- Lenovo has the highest sell for No OS
- Apple has the highest sell for macOS and mac OS X
- Dell has the highest sell for Linux
- Microsoft has the highest sell for Windows 10 S


        """)
        
    with st.expander("Bivariate Analysis for categorical variables Company vs Type Name"):
        st.image("./Laptop_price_predictions/image/no24.png")
        st.write("""
        From this, we can say that:
- Many companies sell type of product is Notebook
- HP company is the highest sell for type Notebook and Lenovo company is the second highest 
sell for type Notebook after HP company
- Apple, Microsoft, Huawei, Google, and LG company just sell for type Ultrabook
- MSL company is the highest sell for type Gaming
- Dell company is the highest sell for type Ultrabook
- Hp, Acer, Asus, Dell, Lenovo, and Samsung company sell much type of product

        """)

elif option == "Model Building":
    st.write("""## Model Building""")
    st.write()
    st.write("""
    We build some models like Decision Tree, Support Vector Machine,
    Lasso, Ridge, and Linear Regression with cross validation for
    optimization parameter. After that, we did evaluate the model and 
    compare each model like this.
    """)
    st.image("./Laptop_price_predictions/image/no25.png")
    st.write(f"""
    As we can see, the best model is Decision Tree. The evaluation of the model is
    \n
    Training Score : {training_score*100:.1f} %\n
    Test Score : {test_score*100:.1f} %\n
    Mean Absolute Error : {mae*100:.1f} %\n
    Mean Squared Error : {mse*100:.1f} %\n
    Root Mean Squared Error : {rmse*100:.1f} %\n
    Mean Absolute Percentage Error : {mape*100:.1f} %\n
    Some explanation about this model get high like below:
    """)
    st.image("./Laptop_price_predictions/image/no26.png")

elif option == "Predict":
    st.write("""## Predict""")
    st.write()
    results = []
    le = LabelEncoder()
    for i in df.columns[:-1]:
        if i == "Weight" or i == "Inches":
            inputt = st.number_input(i)
        else: 
            inputt = st.text_input(i)
        
        results.append(inputt)
    
    
    a = st.button("Predict")
    if a:
        for i in range(len(df["Weight"])):
            df2.loc[i, "Weight"] = df2.loc[i, "Weight"][:-2]
        df2.loc[:, "Weight"] = df2.loc[:, "Weight"].astype("float")

        for i in results:
            if type(i) == str:
                df2[df.columns[results.index(i)]] = le.fit_transform(df2[df.columns[results.index(i)]])
                results[results.index(i)] = le.transform([i])[0]

        for i in df2.columns:
            if i not in df.columns:
                df2.drop(labels=i, inplace=True, axis=1)

        scaler = StandardScaler().fit(df2.iloc[:, :-1])
        scaler2 = StandardScaler().fit(df2.iloc[:, -1].values.reshape(-1, 1))
        results = scaler.transform(np.array(results).reshape(1, -1))
        target = model.predict(results)
        results = scaler2.inverse_transform([target])
        st.write(results)
