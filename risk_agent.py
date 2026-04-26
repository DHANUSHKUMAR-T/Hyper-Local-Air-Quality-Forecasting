import pandas as pd

class RiskAssessmentAgent:
    """
    Agent responsible for converting PM2.5 to AQI and generating personalized alerts.
    """

    def calculate_aqi(self, pm25):
        """Convert PM2.5 to Indian AQI (simplified approximation)."""
        if pm25 <= 30:
            return (50 / 30) * pm25
        elif pm25 <= 60:
            return 50 + (50 / 30) * (pm25 - 30)
        elif pm25 <= 90:
            return 100 + (100 / 30) * (pm25 - 60)
        elif pm25 <= 120:
            return 200 + (100 / 30) * (pm25 - 90)
        elif pm25 <= 250:
            return 300 + (100 / 130) * (pm25 - 120)
        else:
            return 400 + (100 / 100) * (pm25 - 250)

    def get_risk_level(self, aqi):
        """Assign risk level based on AQI."""
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else: return "Severe"

    def generate_alerts(self, pm25, aqi, risk_level):
        """Generate personalized health alerts."""
        alerts = {
            "General": "No health impacts expected." if risk_level in ["Good", "Satisfactory"] else "Consider reducing heavy exertion.",
            "Asthmatic": "No precautions needed." if risk_level == "Good" else "Keep rescue inhaler nearby; avoid outdoor activity.",
            "Children": "Safe to play outdoors." if risk_level == "Good" else "Limit long outdoor play sessions."
        }
        
        if risk_level == "Moderate":
            alerts["General"] = "Sensitive people may experience slight discomfort."
        elif risk_level == "Poor":
            alerts["General"] = "Health alert: Everyone may begin to experience health effects."
        elif risk_level in ["Very Poor", "Severe"]:
            alerts["General"] = "Health warning: Everyone may experience more serious health effects."
            
        return alerts

    def process_data(self, df):
        """Append AQI and risk levels to a dataframe."""
        df['aqi'] = df['pm25_calibrated_xgb'].apply(self.calculate_aqi)
        df['risk_level'] = df['aqi'].apply(self.get_risk_level)
        return df

if __name__ == "__main__":
    agent = RiskAssessmentAgent()
    pm = 85
    aqi = agent.calculate_aqi(pm)
    risk = agent.get_risk_level(aqi)
    alerts = agent.generate_alerts(pm, aqi, risk)
    
    print(f"PM2.5: {pm} | AQI: {aqi:.0f} | Risk: {risk}")
    print("Alerts:")
    for k, v in alerts.items():
        print(f" - {k}: {v}")
