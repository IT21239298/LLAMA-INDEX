import time
import os
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DnataTravelInsurance:
    def __init__(self, headless=True):
        # Set up Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")  # Updated headless flag
            chrome_options.add_argument("--disable-gpu")   # Often needed with headless
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        
        # Initialize webdriver
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        self.driver.implicitly_wait(10)
        self.wait = WebDriverWait(self.driver, 20)
        self.driver.maximize_window()
        
    def navigate_to_insurance_page(self):
        self.driver.get("https://www.dnatatravel.com/v2/insurance")
        time.sleep(3)
        
        # Handle cookies popup
        try:
            cookies_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept all cookies') or contains(text(), 'Accept All Cookies')]"))
            )
            cookies_button.click()
            time.sleep(2)
        except:
            pass
            
    def select_policy_type(self, policy_type="Single Trip"):
        try:
            policy_dropdown = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "select[ng-model='policyType']"))
            )
            select = Select(policy_dropdown)
            select.select_by_visible_text(policy_type)
            time.sleep(1)
        except:
            pass
            
    def select_travel_destination(self, destination="Europe"):
        try:
            destination_dropdown = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "select[ng-model='travelDestination']"))
            )
            select = Select(destination_dropdown)
            select.select_by_visible_text(destination)
            time.sleep(1)
        except:
            pass
            
    def select_travel_dates(self, start_date=None, end_date=None):
        if not start_date or not end_date:
            today = datetime.now()
            default_start = today + timedelta(days=7)
            default_end = today + timedelta(days=14)
            start_date = default_start.strftime('%d-%m-%Y')
            end_date = default_end.strftime('%d-%m-%Y')
        
        try:
            # Parse the input dates
            try:
                start_parsed = datetime.strptime(start_date, '%d-%m-%Y')
                end_parsed = datetime.strptime(end_date, '%d-%m-%Y')
            except ValueError:
                today = datetime.now()
                start_parsed = today + timedelta(days=7)
                end_parsed = today + timedelta(days=14)
                start_date = start_parsed.strftime('%d-%m-%Y')
                end_date = end_parsed.strftime('%d-%m-%Y')
            
            # Try direct JavaScript method
            try:
                # Set start date
                start_date_field = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input.datepicker-wrapper[placeholder='Start Date']"))
                )
                start_date_field.click()
                time.sleep(2)
                
                self.driver.execute_script(f"arguments[0].value = '{start_date}';", start_date_field)
                self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'))", start_date_field)
                time.sleep(2)
                
                # Set end date
                end_date_field = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input.datepicker-wrapper[placeholder='End date']"))
                )
                end_date_field.click()
                time.sleep(2)
                
                self.driver.execute_script(f"arguments[0].value = '{end_date}';", end_date_field)
                self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'))", end_date_field)
                time.sleep(2)
                
                # Click somewhere else to close datepicker
                self.driver.find_element(By.CSS_SELECTOR, ".search-panel-parts").click()
                time.sleep(1)
                
                # Verify the dates were entered correctly
                start_value = self.driver.execute_script("return arguments[0].value;", start_date_field)
                end_value = self.driver.execute_script("return arguments[0].value;", end_date_field)
                
                # If values don't match, try calendar method
                if start_date not in start_value or end_date not in end_value:
                    raise Exception("Date values don't match expected values")
                    
            except Exception:
                # Try calendar method
                # Clear and click start date
                start_date_field = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input.datepicker-wrapper[placeholder='Start Date']"))
                )
                start_date_field.clear()
                start_date_field.click()
                time.sleep(2)
                
                # Select the date in the calendar
                start_day = start_parsed.day
                day_element = self.driver.find_element(By.XPATH, 
                    f"//td[@data-handler='selectDay']/a[text()='{start_day}']")
                day_element.click()
                time.sleep(2)
                
                # Clear and click end date
                end_date_field = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input.datepicker-wrapper[placeholder='End date']"))
                )
                end_date_field.clear()
                end_date_field.click()
                time.sleep(2)
                
                # Select the date in the calendar
                end_day = end_parsed.day
                day_element = self.driver.find_element(By.XPATH, 
                    f"//td[@data-handler='selectDay']/a[text()='{end_day}']")
                day_element.click()
                time.sleep(2)
                
        except:
            pass
            
    def select_passengers(self, adults=1, children=0, adult_ages=None):
        try:
            # Click on passengers field to open the dropdown
            passengers_field = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".pseudo-input.input-occupancy"))
            )
            passengers_field.click()
            time.sleep(2)
            
            # Select number of adults
            adults_dropdown = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select[ng-model='room.Adults']"))
            )
            adults_select = Select(adults_dropdown)
            adults_select.select_by_visible_text(str(adults))
            time.sleep(2)
            
            # Select number of children
            children_dropdown = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select[ng-model='room.ChildAges.length']"))
            )
            children_select = Select(children_dropdown)
            children_select.select_by_visible_text(str(children))
            time.sleep(2)
            
            # Set adult ages if provided
            if adult_ages and len(adult_ages) > 0:
                adult_age_dropdowns = self.driver.find_elements(By.CSS_SELECTOR, "select.age-selector")
                if len(adult_age_dropdowns) >= len(adult_ages):
                    for i, age in enumerate(adult_ages):
                        age_select = Select(adult_age_dropdowns[i])
                        age_select.select_by_visible_text(str(age))
                        time.sleep(1)
            
            # Click Done button
            done_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".guest-picker-footer button"))
            )
            done_button.click()
            time.sleep(2)
        except:
            pass
            
    def select_policy_extras(self, golf_equipment=False, winter_sports=False):
        try:
            checkboxes = self.driver.find_elements(By.CSS_SELECTOR, ".search-unit-field.policy-extras .checkbox-btn")
            
            # Process Golf Equipment checkbox
            if golf_equipment and len(checkboxes) > 0:
                for checkbox in checkboxes:
                    if "Golf Equipment" in checkbox.text and "checked" not in checkbox.get_attribute("class"):
                        checkbox.click()
                        time.sleep(1)
                        break
            
            # Process Winter Sports checkbox
            if winter_sports and len(checkboxes) > 0:
                for checkbox in checkboxes:
                    if "Winter Sports" in checkbox.text and "checked" not in checkbox.get_attribute("class"):
                        checkbox.click()
                        time.sleep(1)
                        break
        except:
            pass
            
    def search_insurance(self):
        try:
            search_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".search-panel-part.submit button"))
            )
            search_button.click()
            
            # Wait for results page to load
            time.sleep(10)
            
            # Get the current URL which should be the results page
            result_url = self.driver.current_url
            return result_url
        except:
            return None
            
    def close_browser(self):
        self.driver.quit()
            
    def get_insurance_url(self, config):
        """
        Run the process and return only the final URL
        Args:
            config (dict): Configuration dictionary with all parameters
        Returns:
            str: The final URL from the insurance search
        """
        try:
            self.navigate_to_insurance_page()
            self.select_policy_type(config.get('policy_type', 'Single Trip'))
            self.select_travel_destination(config.get('destination', 'Europe'))
            self.select_travel_dates(config.get('start_date'), config.get('end_date'))
            self.select_passengers(
                config.get('adults', 1), 
                config.get('children', 0),
                config.get('adult_ages', [35])
            )
            self.select_policy_extras(
                config.get('golf_equipment', False),
                config.get('winter_sports', False)
            )
            
            result_url = self.search_insurance()
            return result_url
        finally:
            if config.get('close_browser', True):
                self.close_browser()


def get_travel_insurance_url(
    policy_type='Single Trip',
    destination='Europe',
    start_date='21-03-2025',
    end_date='28-03-2025',
    adults=2,
    children=0,
    adult_ages=[35, 40],
    golf_equipment=True,
    winter_sports=False,
    headless=True
):
    """
    Simple function to get the insurance URL with the specified parameters
    Returns:
        str: The final URL from the insurance search
    """
    config = {
        'policy_type': policy_type,
        'destination': destination,
        'start_date': start_date,
        'end_date': end_date,
        'adults': adults,
        'children': children,
        'adult_ages': adult_ages,
        'golf_equipment': golf_equipment,
        'winter_sports': winter_sports,
        'close_browser': True
    }
    
    dnata = DnataTravelInsurance(headless=headless)
    return dnata.get_insurance_url(config)


# Example usage
if __name__ == "__main__":
    print("Getting travel insurance URL (headless mode)...")
    url = get_travel_insurance_url()
    
    if url:
        print(f"Success! Final URL: {url}")
    else:
        print("Failed to get results URL.")