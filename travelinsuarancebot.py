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
import openai  # Using updated OpenAI client

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

class DnataTravelInsurance:
    def __init__(self):
        # Set up Chrome options
        chrome_options = Options()
        # Uncomment the line below if you want to run headless (no browser UI)
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Add additional options to help with date picker issues
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        
        # Initialize webdriver
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        # Set implicit wait time
        self.driver.implicitly_wait(10)
        
        # Set explicit wait with longer timeout
        self.wait = WebDriverWait(self.driver, 20)
        
        # Maximize window to ensure all elements are visible
        self.driver.maximize_window()
        
    def navigate_to_insurance_page(self):
        """Navigate to the insurance page and handle any popups"""
        self.driver.get("https://www.dnatatravel.com/v2/insurance")
        print("Navigated to insurance page")
        time.sleep(3)  # Allow page to fully load
        
        # Handle cookies popup if it appears
        try:
            cookies_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept all cookies') or contains(text(), 'Accept All Cookies')]"))
            )
            cookies_button.click()
            print("Clicked 'Accept all cookies' button")
            time.sleep(2)
        except Exception as e:
            print(f"No cookies popup found or unable to click it: {e}")
            
        # Handle any other popups that might appear
        try:
            # Look for common popup close buttons (×, Close, etc.)
            close_buttons = self.driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'Close') or contains(@class, 'close') or contains(text(), '×')]")
            for button in close_buttons:
                if button.is_displayed():
                    button.click()
                    print("Closed a popup")
                    time.sleep(1)
        except Exception as e:
            print(f"No popups found or unable to close them: {e}")
        
    def select_policy_type(self, policy_type="Single Trip"):
        """
        Select the policy type
        Args:
            policy_type (str): Either "Single Trip" or "Annual"
        """
        try:
            # Wait for the policy type dropdown to be clickable
            policy_dropdown = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "select[ng-model='policyType']"))
            )
            select = Select(policy_dropdown)
            select.select_by_visible_text(policy_type)
            print(f"Selected policy type: {policy_type}")
            time.sleep(1)
        except Exception as e:
            print(f"Error selecting policy type: {e}")
            
    def select_travel_destination(self, destination="Europe"):
        """
        Select the travel destination
        Args:
            destination (str): One of "Europe", "Worldwide (exc USA and Canada)", 
                              "Worldwide", or "MENA"
        """
        try:
            # Wait for the destination dropdown to be clickable
            destination_dropdown = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "select[ng-model='travelDestination']"))
            )
            select = Select(destination_dropdown)
            select.select_by_visible_text(destination)
            print(f"Selected travel destination: {destination}")
            time.sleep(1)
        except Exception as e:
            print(f"Error selecting travel destination: {e}")
            
    def select_travel_dates(self, start_date=None, end_date=None):
        """
        Select the travel dates
        Args:
            start_date (str): Start date in format 'DD-MM-YYYY'
            end_date (str): End date in format 'DD-MM-YYYY'
        """
        # If dates not provided, set default dates (today + 7 days for start, + 14 days for end)
        if not start_date or not end_date:
            today = datetime.now()
            default_start = today + timedelta(days=7)
            default_end = today + timedelta(days=14)
            start_date = default_start.strftime('%d-%m-%Y')
            end_date = default_end.strftime('%d-%m-%Y')
        
        try:
            # Parse the input dates to make sure they're in the correct format
            try:
                start_parsed = datetime.strptime(start_date, '%d-%m-%Y')
                end_parsed = datetime.strptime(end_date, '%d-%m-%Y')
                print(f"Parsed dates - Start: {start_parsed.strftime('%d-%m-%Y')}, End: {end_parsed.strftime('%d-%m-%Y')}")
            except ValueError as e:
                print(f"Error parsing dates: {e}")
                # Use fallback dates
                today = datetime.now()
                start_parsed = today + timedelta(days=7)
                end_parsed = today + timedelta(days=14)
                start_date = start_parsed.strftime('%d-%m-%Y')
                end_date = end_parsed.strftime('%d-%m-%Y')
                print(f"Using fallback dates - Start: {start_date}, End: {end_date}")
            
            # Method 1: Try using the datepicker directly
            try:
                # Click on start date field to open datepicker
                start_date_field = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input.datepicker-wrapper[placeholder='Start Date']"))
                )
                start_date_field.click()
                time.sleep(2)
                
                # Use JavaScript to set the value directly
                self.driver.execute_script(f"arguments[0].value = '{start_date}';", start_date_field)
                # Trigger change event to ensure the date is registered
                self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'))", start_date_field)
                print(f"Set start date using JavaScript: {start_date}")
                time.sleep(2)
                
                # Click on end date field
                end_date_field = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input.datepicker-wrapper[placeholder='End date']"))
                )
                end_date_field.click()
                time.sleep(2)
                
                # Use JavaScript to set the value directly
                self.driver.execute_script(f"arguments[0].value = '{end_date}';", end_date_field)
                # Trigger change event to ensure the date is registered
                self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'))", end_date_field)
                print(f"Set end date using JavaScript: {end_date}")
                time.sleep(2)
                
                # Click somewhere else to close datepicker
                self.driver.find_element(By.CSS_SELECTOR, ".search-panel-parts").click()
                time.sleep(1)
                
                # Verify the dates were entered correctly
                start_value = self.driver.execute_script("return arguments[0].value;", start_date_field)
                end_value = self.driver.execute_script("return arguments[0].value;", end_date_field)
                
                print(f"Verified start date value: {start_value}")
                print(f"Verified end date value: {end_value}")
                
                # If values don't match, try Method 2
                if start_date not in start_value or end_date not in end_value:
                    raise Exception("Date values don't match expected values, trying alternate method")
                    
            except Exception as e:
                print(f"Method 1 failed: {e}")
                print("Trying Method 2: Calendar date selection")
                
                # Method 2: Try selecting dates from the calendar widget
                # First clear any existing dates
                start_date_field = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input.datepicker-wrapper[placeholder='Start Date']"))
                )
                start_date_field.clear()
                start_date_field.click()
                time.sleep(2)
                
                # Extract day, month, year from start_date
                start_day = start_parsed.day
                start_month = start_parsed.month
                start_year = start_parsed.year
                
                # Select the date in the calendar
                # First find and click correct month/year navigation if needed
                # Then click the specific day
                day_element = self.driver.find_element(By.XPATH, 
                    f"//td[@data-handler='selectDay']/a[text()='{start_day}']")
                day_element.click()
                print(f"Selected start day: {start_day}")
                time.sleep(2)
                
                # Now select end date
                end_date_field = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input.datepicker-wrapper[placeholder='End date']"))
                )
                end_date_field.clear()
                end_date_field.click()
                time.sleep(2)
                
                # Extract day, month, year from end_date
                end_day = end_parsed.day
                end_month = end_parsed.month
                end_year = end_parsed.year
                
                # Select the date in the calendar
                day_element = self.driver.find_element(By.XPATH, 
                    f"//td[@data-handler='selectDay']/a[text()='{end_day}']")
                day_element.click()
                print(f"Selected end day: {end_day}")
                time.sleep(2)
            
            # Take a screenshot to verify date selection
            try:
                self.driver.save_screenshot("date_selection.png")
                print("Date selection screenshot saved")
            except:
                print("Could not save date selection screenshot")
                
        except Exception as e:
            print(f"Error selecting travel dates: {e}")
            # Take a screenshot when error occurs
            try:
                self.driver.save_screenshot("date_selection_error.png")
                print("Screenshot saved as date_selection_error.png")
            except:
                print("Could not save screenshot")
            
    def select_passengers(self, adults=1, children=0, adult_ages=None):
        """
        Select the number of passengers and their ages
        Args:
            adults (int): Number of adults
            children (int): Number of children
            adult_ages (list): List of adult ages
        """
        try:
            # Click on passengers field to open the dropdown
            passengers_field = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".pseudo-input.input-occupancy"))
            )
            passengers_field.click()
            print("Clicked on passengers field")
            time.sleep(2)  # Increased wait time
            
            # Select number of adults
            adults_dropdown = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select[ng-model='room.Adults']"))
            )
            
            adults_select = Select(adults_dropdown)
            # Use select_by_visible_text instead of value
            adults_select.select_by_visible_text(str(adults))
            print(f"Selected {adults} adults")
            time.sleep(2)  # Increased wait time
            
            # Select number of children
            children_dropdown = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select[ng-model='room.ChildAges.length']"))
            )
            children_select = Select(children_dropdown)
            children_select.select_by_visible_text(str(children))
            print(f"Selected {children} children")
            time.sleep(2)  # Increased wait time
            
            # Now handle the adult ages - this is critical based on the HTML structure
            if adult_ages and len(adult_ages) > 0:
                # Find all adult age dropdowns - based on the HTML, the selector needs to be more specific
                adult_age_dropdowns = self.driver.find_elements(By.CSS_SELECTOR, "select.age-selector")
                print(f"Found {len(adult_age_dropdowns)} adult age dropdowns")
                
                if len(adult_age_dropdowns) >= len(adult_ages):
                    for i, age in enumerate(adult_ages):
                        # Make sure we're getting a fresh reference to the dropdowns each time
                        # This is important because the DOM might change after each selection
                        age_select = Select(adult_age_dropdowns[i])
                        
                        # Select by visible text rather than value
                        age_select.select_by_visible_text(str(age))
                        print(f"Set adult {i+1} age to {age}")
                        time.sleep(1)
                else:
                    print(f"Warning: Need to set {len(adult_ages)} ages but only found {len(adult_age_dropdowns)} dropdowns")
            
            # Click Done button to close the passenger selection
            done_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".guest-picker-footer button"))
            )
            done_button.click()
            print("Clicked Done button")
            time.sleep(2)  # Increased wait time
        except Exception as e:
            print(f"Error selecting passengers: {e}")
            # Take a screenshot when error occurs to help debug
            try:
                self.driver.save_screenshot("passenger_selection_error.png")
                print("Screenshot saved as passenger_selection_error.png")
            except:
                print("Could not save screenshot")
            
    def select_policy_extras(self, golf_equipment=False, winter_sports=False):
        """
        Select policy extras
        Args:
            golf_equipment (bool): Whether to include golf equipment
            winter_sports (bool): Whether to include winter sports
        """
        try:
            # Find all checkboxes - more specific selector based on HTML structure
            checkboxes = self.driver.find_elements(By.CSS_SELECTOR, ".search-unit-field.policy-extras .checkbox-btn")
            
            # Debug checkbox options
            print(f"Found {len(checkboxes)} policy extra checkboxes")
            for i, checkbox in enumerate(checkboxes):
                label_text = checkbox.text.strip()
                print(f"Checkbox {i+1} text: '{label_text}'")
            
            # Process Golf Equipment checkbox
            if golf_equipment and len(checkboxes) > 0:
                golf_checkbox = None
                
                # Look for the golf equipment checkbox by text content
                for checkbox in checkboxes:
                    if "Golf Equipment" in checkbox.text:
                        golf_checkbox = checkbox
                        break
                
                if golf_checkbox:
                    # Only click if not already checked
                    if "checked" not in golf_checkbox.get_attribute("class"):
                        golf_checkbox.click()
                        print("Selected Golf Equipment")
                        time.sleep(1)
                else:
                    print("Golf Equipment checkbox not found")
            
            # Process Winter Sports checkbox
            if winter_sports and len(checkboxes) > 0:
                winter_checkbox = None
                
                # Look for the winter sports checkbox by text content
                for checkbox in checkboxes:
                    if "Winter Sports" in checkbox.text:
                        winter_checkbox = checkbox
                        break
                
                if winter_checkbox:
                    # Only click if not already checked
                    if "checked" not in winter_checkbox.get_attribute("class"):
                        winter_checkbox.click()
                        print("Selected Winter Sports")
                        time.sleep(1)
                else:
                    print("Winter Sports checkbox not found")
                    
        except Exception as e:
            print(f"Error selecting policy extras: {e}")
            # Take a screenshot when error occurs to help debug
            try:
                self.driver.save_screenshot("policy_extras_error.png")
                print("Screenshot saved as policy_extras_error.png")
            except:
                print("Could not save screenshot")
            
    def search_insurance(self):
        """Click the search button and wait for results"""
        try:
            search_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".search-panel-part.submit button"))
            )
            search_button.click()
            print("Clicked search button")
            
            # Wait for results page to load (this might need adjustment based on the actual behavior)
            time.sleep(5)
            
            # Get the current URL which should be the results page
            result_url = self.driver.current_url
            print(f"Results URL: {result_url}")
            return result_url
        except Exception as e:
            print(f"Error during search: {e}")
            return None
            
    def close_browser(self):
        """Close the browser"""
        self.driver.quit()
        print("Browser closed")
        
    def process_with_openai(self, result_url):
        """
        Process the insurance results using OpenAI API
        Args:
            result_url (str): URL of the insurance results page
        """
        try:
            # Using updated OpenAI client (v1.0.0+)
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Take a screenshot of the results page for reference
            try:
                self.driver.save_screenshot("insurance_results.png")
                print("Saved screenshot of results page as insurance_results.png")
            except:
                print("Could not save screenshot of results page")
            
            # Get the page source to analyze in case URL doesn't have parameters
            page_source = self.driver.page_source
            page_title = self.driver.title
            
            prompt = f"""
            I've just searched for travel insurance on dnatatravel.com and got the following results:
            
            URL: {result_url}
            Page Title: {page_title}
            
            Can you analyze what this represents and if there are any parameters in the URL?
            If there are no parameters in the URL, this might indicate the search hasn't fully completed
            or that the site uses a different mechanism to track search parameters.
            """
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Use appropriate model
                messages=[
                    {"role": "system", "content": "You are an AI assistant analyzing travel insurance search results."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis = response.choices[0].message.content
            print("\nOpenAI Analysis:")
            print(analysis)
            return analysis
        except Exception as e:
            print(f"Error processing with OpenAI: {e}")
            return None
            
    def run_full_process(self, config):
        """
        Run the full process with the provided configuration
        Args:
            config (dict): Configuration dictionary with all parameters
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
            
            if result_url and config.get('use_openai', False):
                self.process_with_openai(result_url)
                
            return result_url
        except Exception as e:
            print(f"Error in full process: {e}")
            return None
        finally:
            if config.get('close_browser', True):
                self.close_browser()

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'policy_type': 'Single Trip',
        'destination': 'Europe',
        'start_date': '21-03-2025',  # DD-MM-YYYY format
        'end_date': '28-03-2025',    # DD-MM-YYYY format
        'adults': 2,
        'children': 0,
        'adult_ages': [35, 40],
        'golf_equipment': True,
        'winter_sports': False,
        'use_openai': True,  # Whether to use OpenAI for analysis
        'close_browser': True  # Whether to close the browser after execution
    }
    
    # Create instance and run
    dnata = DnataTravelInsurance()
    result_url = dnata.run_full_process(config)
    
    if result_url:
        print("\nSuccess! Final results URL:")
        print(result_url)
    else:
        print("\nFailed to get results URL.")