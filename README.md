# Airbnb_NewYork_Analysis

## Project Overview
This project analyzes Airbnb listing data from New York City to explore key factors affecting rental prices and builds predictive models to estimate listing prices accurately. The project includes data exploration, feature engineering, model development, and deployment-ready prediction tools.

## File Structure

Airbnb_NewYork_Analysis/
│
├── Downloads/
│ ├── listings_NewYork.csv # Raw Airbnb listings dataset
│ ├── NewYork.png # Visualization/Image related to dataset or analysis
│ ├── neighbourhood.geojson # GeoJSON file for neighborhood boundaries/maps
│ └── neighbourhood.csv # Additional neighborhood data
│
├── notebooks/ # Jupyter notebooks for EDA, feature engineering, training
├── models/ # Saved trained models (e.g., xgb_price_model.pkl)
├── app/ # Streamlit app scripts for price prediction interface
├── scripts/ # Utility and data processing scripts
└── README.md # Project documentation


## How to Use

1. **Install dependencies**
  pip install -r requirements.txt

2. **Explore the data and models**

Use notebooks in the `notebooks/` directory to understand EDA, feature preparation, and model experiments.

3. **Run the Streamlit app**
  Launch the interactive price prediction app:
    streamlit run app/app.py

4. **Data Files**

- `listings_NewYork.csv` contains the main Airbnb listing data.
- Geographic files (`neighbourhood.geojson` and `neighbourhood.csv`) enable spatial analysis and mapping.
- `NewYork.png` provides illustrative visualizations or static maps used in reports.

## Project Highlights

- Feature engineering includes handling amenities, log-transformations, and encoding locations.
- Models tested range from Linear Regression and SVM to XGBoost and Neural Networks.
- XGBoost demonstrated the strongest predictive power for rental price estimation.
- A fully functional Streamlit app integrates the trained model to enable real-time user predictions.

## Future Work

- Integrate image and text analysis for richer feature sets.
- Implement temporal models to capture seasonal trends.
- Expand to other cities or regions to test model generalizability.
- Deploy the model as a web service for external applications.

## Data Source

Data sourced from the public Airbnb dataset for New York City, originally from [Inside Airbnb](http://insideairbnb.com/get-the-data.html).

---

## Author

**Chirag Choudhary**  
Email: chiragcbsc@gmail.com  
LinkedIn: [linkedin.com/in/yourprofile]([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/chirag-choudhary-64a420265/))  
GitHub: [github.com/yourusername](https://github.com/ChiragHooda)  
Portfolio: [Portfolio]([https://github.com/ChiragHooda](https://chiragcbsc.wixsite.com/chiragaiportfolio-1))  

---


## License

This project is licensed under the MIT License.

---

For questions or contributions, please contact or submit issues via GitHub.
