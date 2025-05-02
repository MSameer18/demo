import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv('ingredients.csv')

# Define RDI ranges for Vitamin B complex (approximate, varies by age/gender)
vitamin_b_rdi = {
    'Vitamin B1': (0.8, 1.2),  # mg
    'Vitamin B11': (200, 400),  # µg (Folate)
    'Vitamin B12': (2, 2.4),  # µg
    'Vitamin B2': (0.9, 1.3),  # mg
    'Vitamin B3': (12, 16),  # mg
    'Vitamin B5': (4, 5),  # mg
    'Vitamin B6': (1.1, 1.7)  # mg
}


# Function to classify a single nutrient value
def classify_nutrient(nutrient, value):
    if pd.isna(value):
        return 'safe'  # Treat missing values as neutral

    # Adjust for unit mismatch: assume data is per serving, scale to per 100g if needed
    # For now, let's assume the data is already in the correct units (per 100g)
    # If units are different, scale accordingly (e.g., multiply Sodium by 1000 if in grams)

    # Sugars (g/100g)
    if nutrient == 'Sugars':
        if value < 10:
            return 'safe'
        elif 10 <= value <= 25:
            return 'moderate'
        else:
            return 'harmful'

    # Sodium (mg/100g)
    elif nutrient == 'Sodium':
        # Data seems to be in grams (e.g., 0.016), convert to mg by multiplying by 1000
        value = value * 1000
        if value < 400:
            return 'safe'
        elif 400 <= value <= 900:
            return 'moderate'
        else:
            return 'harmful'

    # Saturated Fats (g/100g)
    elif nutrient == 'Saturated Fats':
        if value < 5:
            return 'safe'
        elif 5 <= value <= 10:
            return 'moderate'
        else:
            return 'harmful'

    # Fat (Total) (g/100g)
    elif nutrient == 'Fat':
        if value < 10:
            return 'safe'
        elif 10 <= value <= 20:
            return 'moderate'
        else:
            return 'harmful'

    # Cholesterol (mg/100g)
    elif nutrient == 'Cholesterol':
        if value < 75:
            return 'safe'
        elif 75 <= value <= 150:
            return 'moderate'
        else:
            return 'harmful'

    # Carbohydrates (g/100g)
    elif nutrient == 'Carbohydrates':
        if 45 <= value <= 65:
            return 'safe'
        elif 30 <= value < 45 or 65 < value <= 80:
            return 'moderate'
        else:
            return 'harmful'

    # Protein (g/100g)
    elif nutrient == 'Protein':
        if 10 <= value <= 30:
            return 'safe'
        elif 30 < value <= 40:
            return 'moderate'
        else:
            return 'harmful'

    # Dietary Fiber (g/100g)
    elif nutrient == 'Dietary Fiber':
        if value > 3:
            return 'safe'
        elif 1.5 <= value <= 3:
            return 'moderate'
        else:
            return 'harmful'

    # Vitamin A (µg/100g)
    elif nutrient == 'Vitamin A':
        if 500 <= value <= 900:
            return 'safe'
        elif 900 < value <= 1500:
            return 'moderate'
        else:
            return 'harmful'

    # Vitamin C (mg/100g)
    elif nutrient == 'Vitamin C':
        if value > 20:
            return 'safe'
        elif 10 <= value <= 20:
            return 'moderate'
        else:
            return 'harmful'

    # Vitamin D (µg/100g)
    elif nutrient == 'Vitamin D':
        if 5 <= value <= 10:
            return 'safe'
        elif 10 < value <= 25:
            return 'moderate'
        else:
            return 'harmful'

    # Vitamin E (mg/100g)
    elif nutrient == 'Vitamin E':
        if 7 <= value <= 15:
            return 'safe'
        elif 15 < value <= 30:
            return 'moderate'
        else:
            return 'harmful'

    # Vitamin K (µg/100g)
    elif nutrient == 'Vitamin K':
        if 60 <= value <= 120:
            return 'safe'
        elif 120 < value <= 250:
            return 'moderate'
        else:
            return 'harmful'

    # Vitamin B Complex
    elif nutrient in vitamin_b_rdi:
        low, high = vitamin_b_rdi[nutrient]
        if low <= value <= high:
            return 'safe'
        elif (low * 0.5 <= value < low) or (high < value <= high * 1.5):
            return 'moderate'
        else:
            return 'harmful'

    # Calcium (mg/100g)
    elif nutrient == 'Calcium':
        if value > 100:
            return 'safe'
        elif 50 <= value <= 100:
            return 'moderate'
        else:
            return 'harmful'

    # Iron (mg/100g)
    elif nutrient == 'Iron':
        if value > 2:
            return 'safe'
        elif 1 <= value <= 2:
            return 'moderate'
        else:
            return 'harmful'

    # Potassium (mg/100g)
    elif nutrient == 'Potassium':
        if value > 200:
            return 'safe'
        elif 100 <= value <= 200:
            return 'moderate'
        else:
            return 'harmful'

    # Zinc (mg/100g)
    elif nutrient == 'Zinc':
        if value > 1.5:
            return 'safe'
        elif 1 <= value <= 1.5:
            return 'moderate'
        else:
            return 'harmful'

    # Magnesium (mg/100g)
    elif nutrient == 'Magnesium':
        if value > 50:
            return 'safe'
        elif 30 <= value <= 50:
            return 'moderate'
        else:
            return 'harmful'

    # Phosphorus (mg/100g)
    elif nutrient == 'Phosphorus':
        if 100 <= value <= 300:
            return 'safe'
        elif 300 < value <= 500:
            return 'moderate'
        else:
            return 'harmful'

    # Selenium (µg/100g)
    elif nutrient == 'Selenium':
        if 20 <= value <= 55:
            return 'safe'
        elif 55 < value <= 100:
            return 'moderate'
        else:
            return 'harmful'

    return 'safe'  # Default for unclassified nutrients


# Classify all nutrients for each product
nutrients = [
    'Sugars', 'Sodium', 'Saturated Fats', 'Fat', 'Cholesterol', 'Carbohydrates', 'Protein',
    'Dietary Fiber', 'Vitamin A', 'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K',
    'Vitamin B1', 'Vitamin B11', 'Vitamin B12', 'Vitamin B2', 'Vitamin B3', 'Vitamin B5', 'Vitamin B6',
    'Calcium', 'Iron', 'Potassium', 'Zinc', 'Magnesium', 'Phosphorus', 'Selenium'
]

classification_results = {}
for nutrient in nutrients:
    classification_results[nutrient] = data[nutrient].apply(lambda x: classify_nutrient(nutrient, x))

# Combine classifications to determine overall risk level
classification_df = pd.DataFrame(classification_results)


def determine_overall_risk(row):
    classifications = [row[nutrient] for nutrient in nutrients]
    if 'harmful' in classifications:
        return 'harmful'
    elif 'moderate' in classifications:
        return 'moderate'
    else:
        return 'safe'


data['risk_level'] = classification_df.apply(determine_overall_risk, axis=1)


# Calculate health score (adjusted to prevent excessive deductions)
def calculate_health_score(row):
    score = 50  # Start at a neutral score
    for nutrient in nutrients:
        classification = row[nutrient]
        if classification == 'harmful':
            score -= 5  # Reduced penalty to prevent score from dropping too low
        elif classification == 'moderate':
            score -= 2
        elif classification == 'safe':
            score += 3
    return max(0, min(100, score))


data['health_score'] = classification_df.apply(calculate_health_score, axis=1)

# Display more columns in the preview
print("Classified Dataset Preview:")
print(data[nutrients + ['risk_level', 'health_score']].head())

# Train a Random Forest model on the nutritional data
# Prepare the features (nutritional values) and labels (risk_level)
X = data[nutrients].fillna(0)  # Fill missing values with 0 for training
y = data['risk_level'].map({'safe': 0, 'moderate': 1, 'harmful': 2})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")

# Save the model for future use
joblib.dump(model, 'nutritional_risk_classifier.pkl')
print("Trained model saved as 'nutritional_risk_classifier.pkl'")

# Save the classified dataset
data.to_csv('classified_and_scored_food_data.csv', index=False)
print("Classified and scored dataset saved as 'classified_and_scored_food_data.csv'")