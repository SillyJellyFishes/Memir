import unittest
from app.weather import get_weather

class TestWeatherAPI(unittest.TestCase):
    def test_weather_default_city_id(self):
        # Should use London, Ontario, Canada (city_id=6058560)
        result = get_weather()
        self.assertIsInstance(result, dict)
        self.assertIn("weather", result)
        self.assertIn("main", result)
        self.assertEqual(result.get("id"), 6058560)
        desc = result['weather'][0]['description']
        temp = result['main']['temp']
        feels = result['main'].get('feels_like', temp)
        humidity = result['main'].get('humidity', '?')
        print(f"London, ON, CA: {desc}, {temp}°C (feels like {feels}°C), humidity {humidity}%")

    def test_weather_explicit_city_id(self):
        result = get_weather(city_id=6058560)
        self.assertIsInstance(result, dict)
        self.assertIn("weather", result)
        self.assertIn("main", result)
        self.assertEqual(result.get("id"), 6058560)

    def test_weather_invalid_city(self):
        result = get_weather("ThisCityDoesNotExist12345", city_id=None)
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
