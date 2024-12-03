import os
import time
import traceback
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from enum import Enum
import openai
from openai import AsyncOpenAI
import logging
import asyncio
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from contextlib import contextmanager

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OpenAI API key not found in environment variables")
else:
    logging.info("OpenAI API key loaded successfully")

# Article data model
class Article(BaseModel):
    source: str
    original_title: str
    korean_title: str
    link: str
    original_content: Optional[str] = None
    korean_content: Optional[str] = None

# Enum for news sources
class NewsSource(str, Enum):
    INSURANCE_BUSINESS = "Insurance Business"
    INSURANCE_JOURNAL = "Insurance Journal"
    REINSURANCE_NEWS = "Reinsurance News"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper.log')
    ]
)

class ServerConnectionManager:
    def __init__(self, max_retries=5, backoff_factor=0.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = None

    @contextmanager
    def get_session(self):
        try:
            if not self.session:
                retry_strategy = Retry(
                    total=self.max_retries,
                    backoff_factor=self.backoff_factor,
                    status_forcelist=[500, 502, 503, 504, 404],
                    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session = requests.Session()
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                self.session = session
            yield self.session
        except Exception as e:
            logging.error(f"Session error: {str(e)}")
            if self.session:
                self.session.close()
            self.session = None
            raise
        finally:
            if self.session:
                self.session.close()
                self.session = None

class RetryableWebDriver:
    def __init__(self, max_retries=3, backoff_factor=0.5, timeout=20):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.driver = None
        self.connection_retry_count = 0
        self.max_connection_retries = 5

    def restart_driver(self):
        """Improved driver restart logic with proper cleanup"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logging.warning(f"Error while quitting driver: {str(e)}")
            finally:
                self.driver = None

        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-software-rasterizer")  # Add this
        chrome_options.page_load_strategy = 'eager'  # Add this
        
        driver_path = "/Users/kwonjia/.wdm/drivers/chromedriver/mac64/131.0.6778.85/chromedriver-mac-arm64/chromedriver"
        service = Service(driver_path)
        
        try:
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            self.driver.implicitly_wait(self.timeout // 2)  # Shorter implicit wait
            
            # Add connection validation
            self.driver.get("about:blank")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize WebDriver: {str(e)}")
            return False

    def execute_with_retry(self, action, *args, **kwargs):
        last_exception = None
        success = False
        
        for attempt in range(self.max_retries):
            try:
                if not self.driver or not self.is_driver_alive():
                    if not self.restart_driver():
                        raise WebDriverException("Failed to restart driver")
                
                result = action(*args, **kwargs)
                success = True
                return result
                
            except Exception as e:
                last_exception = e
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    time.sleep(wait_time)
                    
                    if isinstance(e, (WebDriverException, ConnectionError)):
                        self.restart_driver()
        
        if not success:
            logging.error(f"All retry attempts failed: {str(last_exception)}")
            raise last_exception

    def is_driver_alive(self):
        """Improved driver health check"""
        try:
            if not self.driver:
                return False
                
            # Multiple checks for driver health
            self.driver.current_url  # Basic connectivity check
            self.driver.window_handles  # Session check
            return True
            
        except Exception:
            return False

    def quit(self):
        """Enhanced cleanup method"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logging.error(f"Error during driver cleanup: {str(e)}")
            finally:
                self.driver = None

class NewsTranslator:
    def __init__(self):
        self.system_prompt = "You are a professional insurance industry translator. Translate the following English insurance industry texts to Korean."
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    async def translate_text(self, text: str) -> str:
        if not text:
            return ""
            
        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text}
                    ],
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                )
                
                # 새로운 API 방식으로 응답 처리
                translated_text = response.choices[0].message.content
                return translated_text.strip()
                
            except Exception as e:
                if attempt == 2:  # Last attempt
                    logging.error(f"Translation error after 3 attempts: {str(e)}")
                    return text
                    
                wait_time = 0.5 * (2 ** attempt)
                logging.warning(f"Translation attempt {attempt + 1} failed, retrying in {wait_time} seconds... Error: {str(e)}")
                await asyncio.sleep(wait_time)

class InsuranceNewsScraper:
    def __init__(self):
        self.retryable_driver = RetryableWebDriver()
        self.driver = self.setup_webdriver()
        self.translator = NewsTranslator()
        self.news_sources = {
            "Insurance Business": {
                "url": "https://www.insurancebusinessmag.com/us/news/breaking-news/",
                "title_selector": (
                    "h2.article-list__head-title a, "
                    "h3.article-list__item-title a"
                ),
                "link_selector": (
                    "h2.article-list__head-title a, "
                    "h3.article-list__item-title a"
                ),
                "content_selector": (
                    "div.article-list__head-summary, "
                    "div.article-list__item-summary"
                ),
            },
            
            "Insurance Journal": {
                "url": "https://www.insurancejournal.com/news/",
                "title_selector": "div.popular-title",
                "link_selector": "a.popular-item",
                "content_selector": "div.entry-meta",
            },
            
            "Reinsurance News": {
                "url": "https://www.reinsurancene.ws/",
                "title_selector": (
                    "h2.card-title a, "
                    "h4.card-title a"
                ),
                "link_selector": (
                    "h2.card-title a, "
                    "h4.card-title a"
                ),
                "content_selector": (
                    "p.article-list__head-summary, "
                    "p.article-list__item-summary"
                ),
            },
        }

    def setup_webdriver(self):
        chrome_options = Options()
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")

        driver_path = "/Users/kwonjia/.wdm/drivers/chromedriver/mac64/131.0.6778.85/chromedriver-mac-arm64/chromedriver"
        service = Service(driver_path)

        return self.retryable_driver.execute_with_retry(
            webdriver.Chrome,
            service=service,
            options=chrome_options
        )

    def cleanup(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logging.error(f"Error during cleanup: {str(e)}")

    async def scrape_news(self, source_name: str, limit: int = 20) -> List[Article]:
        source_info = self.news_sources.get(source_name)
        if not source_info:
            raise HTTPException(status_code=404, detail="Source not found")

        articles = []
        try:
            def get_page():
                self.driver.get(source_info["url"])
                time.sleep(random.uniform(2, 4))
                return BeautifulSoup(self.driver.page_source, "html.parser")

            soup = self.retryable_driver.execute_with_retry(get_page)

            title_elements = soup.select(source_info["title_selector"])[:limit]
            link_elements = soup.select(source_info["link_selector"])[:limit]

            for title_elem, link_elem in zip(title_elements, link_elements):
                try:
                    title = title_elem.get_text().strip()
                    link = link_elem.get("href")
                    if not link.startswith("http"):
                        link = f"https://{link.lstrip('/')}"

                    def get_article_content():
                        self.driver.get(link)
                        time.sleep(random.uniform(1, 3))
                        content_soup = BeautifulSoup(self.driver.page_source, "html.parser")
                        content_elem = content_soup.select_one(source_info["content_selector"])
                        return content_elem.get_text().strip() if content_elem else ""

                    content = self.retryable_driver.execute_with_retry(get_article_content)

                    korean_title = await self.translator.translate_text(title)
                    korean_content = await self.translator.translate_text(content) if content else None

                    article = Article(
                        source=source_name,
                        original_title=title,
                        korean_title=korean_title,
                        link=link,
                        original_content=content,
                        korean_content=korean_content,
                    )
                    articles.append(article)
                    
                except Exception as e:
                    logging.error(f"Error processing article: {str(e)}")
                    continue

            return articles
            
        except Exception as e:
            logging.error(f"Error scraping {source_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error scraping {source_name}: {str(e)}")
            
        finally:
            self.cleanup()

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Insurance News API",
        "version": "1.0.0",
        "endpoints": {
            "/": "This help message",
            "/news/{source}": "Get news from specific source",
        },
    }

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Application shutdown initiated")
    # 활성 WebDriver 세션들 정리
    for scraper in active_scrapers:
        try:
            scraper.cleanup()
        except:
            pass
    logging.info("Application shutdown completed")

# 전역 변수로 활성 스크래퍼 추적
active_scrapers = set()

@app.get("/news/{source}", response_model=List[Article])
async def get_news(source: NewsSource, limit: Optional[int] = 20):
    scraper = None
    try:
        scraper = InsuranceNewsScraper()
        active_scrapers.add(scraper)
        return await scraper.scrape_news(source.value, limit)
    except Exception as e:
        logging.error(f"Error in get_news endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if scraper:
            scraper.cleanup()
            active_scrapers.remove(scraper)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout"
                },
            },
            "root": {
                "handlers": ["default"],
                "level": "INFO",
            },
        }
    )