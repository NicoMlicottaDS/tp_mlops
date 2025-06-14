import requests

url = "http://localhost:8000/predict"
data = {
    "Make": "Ford",
    "Model": "Fiesta",
    "Year": 2015,
    "Engine_HP": 120,
    "Engine_Cylinders": 4,
    "Transmission_Type": "MANUAL",
    "Driven_Wheels": "front wheel drive",
    "Number_of_Doors": 4,
    "Market_Category": None,
    "Vehicle_Size": "Compact",
    "Vehicle_Style": "Sedan",
    "highway_MPG": 36,
    "city_MPG": 28,
    "Popularity": 138
}

response = requests.post(url, json=data)
print("Status:", response.status_code)
print("Response:", response.json())
