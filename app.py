import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load('xgb_price_model.pkl')

# Get the exact feature names expected by your model
expected_features = [
    'estimated_revenue_l365d_log', 'number_of_reviews_ltm_log', 'room_type_Entire home/apt', 
    'minimum_nights_avg_ntm_capped', 'availability_30', 'accommodates_log', 'recency_last_review', 
    'bedrooms_log', 'calculated_host_listings_count_private_rooms', 'dishwasher', 
    'neighbourhood_Midtown', 'calculated_host_listings_count_entire_homes_log', 'reviews_per_month_log', 
    'review_span_days', 'number_of_reviews', 'neighbourhood_Murray Hill', 'beds_log', 'washer', 
    'gym', 'neighbourhood_Upper East Side', "neighbourhood_Hell's Kitchen", 'neighbourhood_East Village', 
    'instant_bookable_f', 'room_type_Private room', 'neighbourhood_Chelsea', 'neighbourhood_West Village', 
    'neighbourhood_Williamsburg', 'neighbourhood_Upper West Side', 'hair_dryer', 'indoor_fireplace', 
    'air_conditioning', 'microwave', 'pool', 'neighbourhood_Greenwich Village', 
    'neighbourhood_Long Island City', 'neighbourhood_SoHo', 'neighbourhood_Longwood', 
    'neighbourhood_Crown Heights', 'neighbourhood_Lower East Side', 'neighbourhood_Jamaica', 
    'neighbourhood_Fieldston', 'neighbourhood_Tribeca', 'hot_water', 'refrigerator', 
    'neighbourhood_Bushwick', 'neighbourhood_East New York', 'coffee_maker'
]

st.title("Airbnb Price Predictor")
st.markdown("Enter your listing details to get a price estimate using our trained XGBoost model.")

# User inputs
st.header("Listing Details")

# Room Type
room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room"])

# Accommodation details
accommodates = st.number_input("Number of guests", min_value=1, max_value=20, value=2)
bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10, value=1)
beds = st.number_input("Number of beds", min_value=1, max_value=15, value=1)

# Minimum nights and availability
minimum_nights = st.number_input("Minimum nights", min_value=1, max_value=365, value=1)
availability_30 = st.number_input("Availability in next 30 days", min_value=0, max_value=30, value=15)

# Neighbourhood selection (NYC neighbourhoods from your model)
neighbourhood = st.selectbox("Neighbourhood", [
    "None", "Midtown", "Murray Hill", "Upper East Side", "Hell's Kitchen", "East Village", 
    "Chelsea", "West Village", "Williamsburg", "Upper West Side", "Greenwich Village", 
    "Long Island City", "SoHo", "Longwood", "Crown Heights", "Lower East Side", 
    "Jamaica", "Fieldston", "Tribeca", "Bushwick", "East New York"
])

# Amenities
st.subheader("Amenities Available")
col1, col2, col3 = st.columns(3)

with col1:
    dishwasher = st.checkbox("Dishwasher")
    washer = st.checkbox("Washer")
    gym = st.checkbox("Gym")
    hair_dryer = st.checkbox("Hair Dryer")

with col2:
    indoor_fireplace = st.checkbox("Indoor Fireplace")
    air_conditioning = st.checkbox("Air Conditioning")
    microwave = st.checkbox("Microwave")
    pool = st.checkbox("Pool")

with col3:
    hot_water = st.checkbox("Hot Water")
    refrigerator = st.checkbox("Refrigerator")
    coffee_maker = st.checkbox("Coffee Maker")

# Additional features (set reasonable defaults)
instant_bookable = st.checkbox("Instant Bookable", value=False)

# Review related inputs
number_of_reviews = st.number_input("Total number of reviews", min_value=0, max_value=1000, value=10)
reviews_per_month = st.number_input("Reviews per month", min_value=0.0, max_value=20.0, value=1.0)
review_span_days = st.number_input("Days since first review", min_value=0, max_value=5000, value=365)
recency_last_review = st.number_input("Days since last review", min_value=0, max_value=1000, value=30)

# Host listings
host_entire_homes = st.number_input("Host's total entire homes", min_value=0, max_value=500, value=1)
host_private_rooms = st.number_input("Host's total private rooms", min_value=0, max_value=500, value=0)

if st.button("Predict Price"):
    # Create DataFrame with all expected features initialized to 0
    X = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # Set the actual values
    # Log transformations for numerical features
    X['accommodates_log'] = np.log(accommodates)
    X['bedrooms_log'] = np.log(bedrooms)
    X['beds_log'] = np.log(beds)
    
    # Reviews and availability
    X['availability_30'] = availability_30
    X['minimum_nights_avg_ntm_capped'] = minimum_nights
    X['number_of_reviews'] = number_of_reviews
    X['reviews_per_month_log'] = np.log(reviews_per_month + 1)  # +1 to avoid log(0)
    X['number_of_reviews_ltm_log'] = np.log(number_of_reviews + 1)
    X['review_span_days'] = review_span_days
    X['recency_last_review'] = recency_last_review
    
    # Room type
    if room_type == "Entire home/apt":
        X['room_type_Entire home/apt'] = 1
    elif room_type == "Private room":
        X['room_type_Private room'] = 1
    
    # Neighbourhood
    if neighbourhood != "None":
        neighbourhood_col = f'neighbourhood_{neighbourhood}'
        if neighbourhood_col in X.columns:
            X[neighbourhood_col] = 1
    
    # Amenities
    X['dishwasher'] = 1 if dishwasher else 0
    X['washer'] = 1 if washer else 0
    X['gym'] = 1 if gym else 0
    X['hair_dryer'] = 1 if hair_dryer else 0
    X['indoor_fireplace'] = 1 if indoor_fireplace else 0
    X['air_conditioning'] = 1 if air_conditioning else 0
    X['microwave'] = 1 if microwave else 0
    X['pool'] = 1 if pool else 0
    X['hot_water'] = 1 if hot_water else 0
    X['refrigerator'] = 1 if refrigerator else 0
    X['coffee_maker'] = 1 if coffee_maker else 0
    
    # Other features
    X['instant_bookable_f'] = 1 if instant_bookable else 0
    X['calculated_host_listings_count_entire_homes_log'] = np.log(host_entire_homes + 1)
    X['calculated_host_listings_count_private_rooms'] = host_private_rooms
    
    # Set estimated revenue (you may need to calculate this or use a default)
    X['estimated_revenue_l365d_log'] = np.log(100)  # Default value, adjust as needed
    
    try:
        # Make prediction
        prediction_log = model.predict(X)[0]
        predicted_price = np.exp(prediction_log)  # Assuming log-transformed target
        
        st.success(f"**Estimated Price per Night: ${predicted_price:.2f}**")
        
        # Show confidence information
        st.info(f"Prediction based on {len(expected_features)} features including location, amenities, and property details.")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please check that all inputs are valid.")

# Show model information
with st.expander("Model Information"):
    st.write(f"This model uses {len(expected_features)} features to predict Airbnb prices.")
    st.write("Features include room type, location, amenities, host information, and review metrics.")
