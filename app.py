import streamlit as st
import pandas as pd
import numpy as np 
import openmeteo_requests
import requests_cache
from retry_requests import retry  
import joblib 
import pickle
import matplotlib.pyplot as plt
import os
import plotly.express as px
from geopy.geocoders import Nominatim


st.write("""
    # Short-term prediction photovoltaic energy
   This interface makes it possible to predict the performance of a photovoltaic system for the next 7 days based on upcoming weather conditions, taking into account the geographic location of the installation.
    """)

with open("Help.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
    
st.sidebar.download_button(label="Download Help Documentation",data=PDFbyte, file_name="Help.pdf",mime='application/octet-stream')

st.sidebar.header("PV technology")
st.title("")
# Create a sidebar navigation menu
selected_page = st.sidebar.selectbox("Select a technology ", ["Silicium cristallin", "CIS", "CdTe"])
#Surface =st.text_input("Surface")



Surface= st.sidebar.text_input("Surface (m²)", "Ex. 16" )
Location = st.text_input("Location", "Ex. 19 Rue Richer, 75009 Paris")  
#latitude = st.text_input("latitude")
#longitude = st.text_input("longitude")




if st.button("Start energy prediction"): 


 
    # calling the Nominatim tool and create Nominatim class
    loc = Nominatim(user_agent="Geopy Library")

   # entering the location name
    getLoc = loc.geocode(Location)

   # printing address
    print(getLoc.address)

   # printing latitude and longitude
    print("Latitude = ", getLoc.latitude, "\n")
    print("Longitude = ", getLoc.longitude)

    df = pd.DataFrame({
      "col1": getLoc.latitude,
      "col2": getLoc.longitude,
      "col3": 60,
      "col4": np.random.rand(1000, 4).tolist(),
    })

    st.map(df,
     latitude='col1',
     longitude='col2',
     size='col3',
     color='col4')
    
    latitude = getLoc.latitude
    longitude = getLoc.longitude
    def main():
     if selected_page == "Silicium cristallin":
         # Load the model from the file
        model_from_joblib = joblib.load('./Silicium_modèle.pkl')

        with open(os.path.join('./','Silicium_modèle.pkl'), 'rb') as f:
           pickle.load(f)

        # Use the loaded model to make predictions
           predict = model_from_joblib.predict(df_hourly_pre)
         
     elif selected_page == "CIS":
         # Load the model from the file
           model_from_joblib = joblib.load('CIS_modèle.pkl')

           with open(os.path.join('..','CIS_modèle.pkl'), 'rb') as f:
            pickle.load(f)

        # Use the loaded model to make predictions
           predict = model_from_joblib.predict(df_hourly_pre)
         
     elif selected_page == "CdTe":
        # Load the model from the file
        model_from_joblib = joblib.load('CdTe_modèle.pkl')

        with open(os.path.join('..','CdTe_modèle.pkl'), 'rb') as f:
            pickle.load(f)

        # Use the loaded model to make predictions
            predict =model_from_joblib.predict(df_hourly_pre)
     return  predict  
         




    


    def user_input():
    
     data= {'Latitude': latitude, 'Longitude': longitude, 'Surface' : Surface }
     paramètres = pd.DataFrame(data,index=[0])

     return paramètres

    df_point = user_input()

    st.subheader('Geographical point & Surface')
    st.write(df_point)

    st.subheader('Weather Forecast ')
    def user():
    # Setup the Open-Meteo API client with cache and retry on error
     cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
     retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
     openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
     url = "https://api.open-meteo.com/v1/forecast"
     params = {
	 "latitude": latitude,
	 "longitude": longitude,
	 "hourly": ["temperature_2m", "direct_normal_irradiance"],
	 "timezone": "Europe/Berlin"
     }
     responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
     response = responses[0]
     print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
     print(f"Elevation {response.Elevation()} m asl")
     print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
     print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

     # Process hourly data. The order of variables needs to be the same as requested.
     hourly = response.Hourly()
     hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
     hourly_direct_radiation = hourly.Variables(1).ValuesAsNumpy()

     hourly_data = {"date": pd.date_range(
	  start = pd.to_datetime(hourly.Time(), unit = "s"),
	  end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
	  freq = pd.Timedelta(seconds = hourly.Interval()),
	  inclusive = "left"
      )}
     hourly_data["temperature_2m"] = hourly_temperature_2m
     hourly_data["direct_normal_irradiance"] = hourly_direct_radiation

     hourly_dataframe = pd.DataFrame(data = hourly_data)
      #print(hourly_dataframe)
     return hourly_dataframe


    df_hourly= user()

    st.write(df_hourly)











    st.subheader('Energy prediction')
    df_hourly_pre = df_hourly.drop('date', axis=1)
  


    predict = main()
    Prediction= pd.DataFrame(predict, columns=['Predicted_Energy'], dtype= float)

    df_final = pd.concat([df_hourly, Prediction.apply(lambda x: x * float(Surface))], axis=1)
    

    st.write(df_final)
    st.header("Visualization")
    data1 = df_final['temperature_2m']
    data2 = df_final['direct_normal_irradiance']
    data3 = df_final['date']
    

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Temperature")
    plt.plot(data3, data1)
    plt.xticks(rotation = 90)
    #plt.xlabel("Dates")
    plt.ylabel("C°")
    plt.grid(True, which="both")
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.title("direct_normal_irradiance")
    plt.plot(data3, data2, "r--")
    plt.xticks(rotation = 90)
    #plt.xlabel("Dates")
    plt.ylabel("W/m²")
    plt.grid(True, which="both")
    plt.tight_layout()

    # Affichage de la figure.
    plt.legend(prop = {'size': 12})
    plt.show()
    st.pyplot(plt)

     
    fig = px.line(df_final, x='date', y='Predicted_Energy', title="P(W) vs Time(Hour)", labels={"date": "date","Predicted_Energy": "Power"})
    fig.update_traces(line_color='green')
    

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    
