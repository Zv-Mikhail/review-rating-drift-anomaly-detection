from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions

class WebDriverManager:
    @staticmethod
    def create_webdriver(headless: bool = False):
        opts = ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        # стабильность и меньше «первого запуска»
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--incognito")
        opts.add_argument("--no-first-run")
        opts.add_argument("--no-default-browser-check")
        opts.add_argument("--disable-extensions")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(60)
        driver.implicitly_wait(8)   # мягкий дефолт, но будем использовать явные ожидания
        return driver
