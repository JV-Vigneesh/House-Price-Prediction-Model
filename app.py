import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model/best_model.keras", compile=False)
preprocessor = joblib.load("model/scaler.pkl")
y_scaler = joblib.load("model/y_scaler.pkl")

df_data = pd.read_csv("house_price_dataset_india_12k.csv")
df_data.columns = df_data.columns.str.strip()

cities = sorted(df_data["City"].unique())

locality_map = {"Budget": 0, "Mid": 1, "Premium": 2}


# 🔥 Indian format
def format_indian(num):
    num = int(num)
    s = str(num)
    last3 = s[-3:]
    rest = s[:-3]
    if rest:
        rest = rest[::-1]
        rest = ','.join([rest[i:i+2] for i in range(0, len(rest), 2)])
        rest = rest[::-1]
        return f"{rest},{last3}"
    return last3


# Lakhs / Crores
def format_short(num):
    if num >= 1e7:
        return f"₹ {num/1e7:.2f} Cr"
    return f"₹ {num/1e5:.2f} Lakhs"


@app.route("/")
def home():
    return render_template(
        "index.html",
        cities=cities,
        form_data=None,      # 🔥 FORCE NONE
        used_advanced=False # 🔥 ADD THIS
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        tier = data["Locality_Tier"]

        # Tier defaults
        if tier == "Budget":
            defaults = dict(metro=5, city_center=20, school=2.5, hospital=3, crime=60, floor=3, total_floors=6)
        elif tier == "Mid":
            defaults = dict(metro=2, city_center=10, school=1.5, hospital=2, crime=35, floor=5, total_floors=10)
        else:
            defaults = dict(metro=1, city_center=5, school=0.5, hospital=1, crime=15, floor=10, total_floors=20)

        super_area = float(data["Super_Area_sqft"])

        # 🔥 FIX: carpet area
        carpet_area = float(data.get("Carpet_Area_sqft") or super_area * 0.8)

        floor_no = float(data.get("Floor_No") or defaults["floor"])
        total_floors = float(data.get("Total_Floors") or defaults["total_floors"])
        metro = float(data.get("Distance_to_Metro_km") or defaults["metro"])
        city_center = float(data.get("Distance_to_CityCenter_km") or defaults["city_center"])
        school = float(data.get("Nearby_School_km") or defaults["school"])
        hospital = float(data.get("Nearby_Hospital_km") or defaults["hospital"])
        crime = float(data.get("Crime_Rate_Index") or defaults["crime"])

        input_dict = {
            "City": data["City"],
            "Locality_Tier": locality_map[tier],
            "BHK": float(data["BHK"]),
            "Bathrooms": float(data["Bathrooms"]),
            "Super_Area_sqft": super_area,
            "Carpet_Area_sqft": carpet_area,
            "Floor_No": floor_no,
            "Total_Floors": total_floors,
            "Property_Age_years": float(data["Property_Age_years"]),
            "Parking": int(data.get("Parking", 1)),
            "Furnishing": data.get("Furnishing", "Semi-Furnished"),
            "Lift": int(data.get("Lift", 1)),
            "Gated_Society": int(data.get("Gated_Society", 1)),
            "Distance_to_Metro_km": metro,
            "Distance_to_CityCenter_km": city_center,
            "Nearby_School_km": school,
            "Nearby_Hospital_km": hospital,
            "Crime_Rate_Index": crime,
        }

        df = pd.DataFrame([input_dict])
        processed = preprocessor.transform(df)

        pred_scaled = model.predict(processed)
        prediction = y_scaler.inverse_transform(pred_scaled)[0][0]

        # 🔥 Formatted outputs
        formatted_full = "₹ " + format_indian(prediction)
        formatted_short = format_short(prediction)

        # 🔥 Range
        rmse = 1237717
        low = max(0, prediction - rmse)
        high = prediction + rmse

        low_str = format_indian(low)
        high_str = format_indian(high)

        # 🔥 Confidence
        confidence = round(max(0, 100 - (rmse / prediction * 100)), 2)

        used_advanced = any([
            data.get("Carpet_Area_sqft"),
            data.get("Floor_No"),
            data.get("Total_Floors"),
            data.get("Distance_to_Metro_km"),
            data.get("Crime_Rate_Index")
        ])

        return render_template(
            "index.html",
            prediction_text=formatted_full,
            short_price=formatted_short,
            low=low_str,
            high=high_str,
            confidence=confidence,
            pred_value=int(prediction),
            cities=cities,
            form_data=data,
            used_advanced=used_advanced
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            cities=cities,
            form_data=data
        )


if __name__ == "__main__":
    app.run(debug=True)