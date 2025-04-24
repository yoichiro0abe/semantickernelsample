from typing import Annotated
from semantic_kernel.functions import kernel_function
import json
import aiohttp
import os
from typing import Annotated
from semantic_kernel.functions import kernel_function
from datetime import datetime, timedelta
import unittest
import asyncio


class LightsPlugin:
    lights = [
        {"id": 1, "name": "Table Lamp", "is_on": False},
        {"id": 2, "name": "Porch light", "is_on": False},
        {"id": 3, "name": "Chandelier", "is_on": True},
    ]

    @kernel_function(
        name="get_lights",
        description="Gets a list of lights and their current state",
    )
    def get_state(
        self,
    ) -> str:
        """Gets a list of lights and their current state."""
        return self.lights

    @kernel_function(
        name="change_state",
        description="Changes the state of the light",
    )
    def change_state(
        self,
        id: int,
        is_on: bool,
    ) -> str:
        """Changes the state of the light."""
        for light in self.lights:
            if light["id"] == id:
                light["is_on"] = is_on
                return light
        return None


class WeatherPlugin:
    """
    A plugin for retrieving weather forecast information from the Japan Meteorological Agency (JMA).
    """

    def __init__(self, area_codes_file="area_codes.json"):
        self.area_codes_url = "https://www.jma.go.jp/bosai/common/const/area.json"
        self.forecast_base_url = "https://www.jma.go.jp/bosai/forecast/data/forecast/"
        self.area_codes_file = area_codes_file
        self.area_codes = {}

    async def load_area_codes(self):
        """
        Loads the area codes from the JMA website or a local file.
        """
        if os.path.exists(self.area_codes_file):
            print("Loading area codes from local file...")
            try:
                with open(self.area_codes_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    name_to_code = {}
    
                    # Process all sections in the JSON data without hardcoding section names
                    for section_name, section_data in json_data.items():
                        # Each section contains area codes as keys and area details as values
                        for area_code, area_details in section_data.items():
                            # Extract the name and add it to our mapping
                            if "name" in area_details:
                                area_name = area_details["name"]
                                name_to_code[area_name] = area_code

  
                    # Merge all sections into self.area_codes
                    self.area_codes = name_to_code

                print("Area codes loaded from file.")
                return
            except json.JSONDecodeError:
                print(
                    "Error: Invalid JSON format in area_codes.json. Fetching from URL..."
                )
            except Exception as e:
                print(
                    f"An error occurred while loading area codes from file: {e}. Fetching from URL..."
                )

        print("Fetching area codes from URL...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.area_codes_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.area_codes.update(data.get("centers", {}))
                        self.area_codes.update(data.get("offices", {}))
                        self.area_codes.update(data.get("class10s", {}))
                        self.area_codes.update(data.get("class15s", {}))
                        self.save_area_codes(data)
                    else:
                        print(f"Failed to load area codes: {response.status}")
        except Exception as e:
            print(f"An error occurred while loading area codes: {e}")

    def save_area_codes(self, data):
        """
        Saves the area codes to a local file.
        """
        try:
            with open(self.area_codes_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Area codes saved to {self.area_codes_file}")
        except Exception as e:
            print(f"An error occurred while saving area codes: {e}")

    async def get_weather_forecast(self, area_name: str, date_str: str) -> str:
        """
        Retrieves the weather forecast for a given area and date.

        Args:
            area_name (str): The name of the area (e.g., "東京").
            date_str (str): The date for which to retrieve the forecast (e.g., "2024-03-15").

        Returns:
            str: The weather forecast information as a string, or an error message.
        """
        area_code = self.find_area_code(area_name)
        if not area_code:
            return f"Error: Area '{area_name}' not found."

        try:
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return "Error: Invalid date format. Please use YYYY-MM-DD."

        forecast_url = f"{self.forecast_base_url}{area_code}.json"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(forecast_url) as response:
                    if response.status == 200:
                        forecast_data = await response.json()
                        return self.extract_forecast(forecast_data, date)
                    else:
                        return f"Error: Failed to retrieve forecast data for {area_name} (code: {area_code}). Status code: {response.status}"
        except Exception as e:
            return f"An error occurred while fetching the forecast: {e}"

    def find_area_code(self, area_name: str) -> str | None:
        """
        Finds the area code for a given area name.

        Args:
            area_name (str): The name of the area.

        Returns:
            str | None: The area code if found, otherwise None.
        """
        return self.area_codes.get(area_name)


    def extract_forecast(self, forecast_data, date) -> str:
        """
        Extracts the weather forecast for a specific date from the forecast data.

        Args:
            forecast_data (dict): The forecast data from the JMA API.
            date (datetime.date): The date for which to extract the forecast.

        Returns:
            str: The weather forecast for the specified date.
        """
        target_date_str = date.strftime("%Y-%m-%d")
        for report in forecast_data:
            for area_forecast in report["timeSeries"]:
                for time_define in area_forecast["timeDefines"]:
                    time_define_date = datetime.strptime(
                        time_define, "%Y-%m-%dT%H:%M:%S%z"
                    ).date()

                    areas = area_forecast['areas']
                    for area in areas:
                        if "weathers" in area:
                            weathers = area["weathers"]
                            if len(weathers) > 0:
                                return (
                                    f"The weather forecast for {target_date_str} is: {weathers[0]}"
                                )
                            else:
                                return f"No weather data found for {target_date_str}"
                        else:
                            return f"No weather data found for {target_date_str}"
        return f"No forecast found for {target_date_str}"

    @kernel_function(
        name="get_weather",
        description="Gets the weather forecast for a specific area and date.",
    )
    async def get_weather(
        self,
        area_name: Annotated[str, "The name of the area (e.g., 東京)"],
        date_str: Annotated[str, "The date for which to retrieve the forecast (e.g., 2024-03-15)"],
    ) -> str:
        """
        Gets the weather forecast for a specific area and date.

        Args:
            area_name (str): The name of the area.
            date_str (str): The date for which to retrieve the forecast.

        Returns:
            str: The weather forecast information.
        """
        print(f"Getting weather forecast for {area_name} on {date_str}...")
        return await self.get_weather_forecast(area_name, date_str)


class CurrentDatePlugin:
    @kernel_function(
        name="get_current_time",
        description="Gets the current time.",
    )
    def get_current_time(self) -> str:
        """Gets the current time."""
        import datetime

        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")


class TestFindAreaCode(unittest.TestCase):
    def setUp(self):
        # Create a dummy area_codes.json for testing
       
        self.weather_plugin = WeatherPlugin()
        asyncio.run(self.weather_plugin.load_area_codes())

    def tearDown(self):
        # Remove the dummy area_codes.json after testing
        print("Cleaning up test data...")


    def test_find_area_code_child(self):
        self.assertEqual(self.weather_plugin.find_area_code("東京地方"), "130010")
        self.assertEqual(self.weather_plugin.find_area_code("大阪府"), "270000")

    def test_find_area_code_grandchild(self):
        self.assertEqual(self.weather_plugin.find_area_code("大阪市"), "2710000")
        self.assertEqual(self.weather_plugin.find_area_code("神戸市"), "2810000")


    def test_find_area_code_not_found(self):
        self.assertIsNone(self.weather_plugin.find_area_code("存在しない場所"))


# Run the tests if the script is executed directly
if __name__ == "__main__":
    unittest.main()
