from openai import OpenAI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
import asyncio
import time
import random
import os
from dotenv import load_dotenv
from enum import Enum
import hashlib
from datetime import datetime, timedelta
import traceback

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class NewsTranslator:
    def __init__(self):
        self.system_prompt = "You are a professional translator. Translate the following English texts to Korean, maintaining the original meaning and nuance while making it natural in Korean. Return translations in the same order as input texts, separated by ||| delimiter."
        self.batch_size = 5  # 한 번의 API 호출로 처리할 텍스트 수
    
    async def translate_batch(self, texts: list) -> list:
        if not texts:
            return []
        
        try:
            combined_text = "\n---\n".join(texts)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": combined_text}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            translations = response.choices[0].message.content.strip().split("|||")
            return [t.strip() for t in translations]
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return texts
    
    async def translate_text(self, text: str) -> str:
        if not text:
            return ""
        try:
            translations = await self.translate_batch([text])
            return translations[0] if translations else text
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text


class Article(BaseModel):
    source: str
    original_title: str
    korean_title: str
    link: str
    date: str
    first_seen: str  
    last_updated: str

class NewsSource(str, Enum):
    ADWEEK = "AdWeek"
    MARKETING_WEEK = "Marketing Week"
    CAMPAIGN = "Campaign"

# FastAPI 앱 초기화
app = FastAPI(
    title="Marketing News API",
    description="Marketing news scraping API from various sources",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 스크래핑 상태를 저장할 전역 변수
scraping_status = {
    "is_running": False,
    "status": "idle",
    "total_articles": 0,
    "sources_completed": [],
    "last_updated": None
}

class MarketingNewsScraper:
    def __init__(self):
        self.setup_webdriver()
        self.translator = NewsTranslator()
        self.news_sources = {
            'AdWeek': {
                'url': 'https://www.adweek.com/category/ad-of-the-day/',
                'title_selector': 'h2.section__title a',
                'link_selector': 'h2.section__title a',
                'content_selector': 'div.article-content'
            },
            'Marketing Week': {
                'url': 'https://www.marketingweek.com/marketing-news/',
                'title_selector': 'h2.hentry-title a',
                'link_selector': 'h2.hentry-title a',
                'content_selector': 'div.article-body'
            },
            'Campaign': {
                'url': 'https://www.campaignlive.com/us/just-published-on-campaign/creativity-news',
                'title_selector': '.storyContent h3 a',
                'link_selector': '.storyContent h3 a',
                'content_selector': 'div.articleText'
            }
        }
        self.data_dir = 'scraped_data'
        os.makedirs(self.data_dir, exist_ok=True)

    def setup_webdriver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument("--remote-debugging-port=9222")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        chrome_options.page_load_strategy = 'eager'
        
        try:
            # 새로 다운로드한 ChromeDriver 경로 지정
            driver_path = os.path.expanduser("~/webdrivers/chromedriver-mac-arm64/chromedriver")
            service = Service(
                executable_path=driver_path,
                log_path=os.devnull
            )
            
            self.driver = webdriver.Chrome(
                service=service, 
                options=chrome_options
            )
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(30)
        except Exception as e:
            print(f"WebDriver 초기화 오류: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    def cleanup(self):
        if hasattr(self, 'driver'):
            self.driver.quit()

    async def get_page_content(self, url: str) -> str:
        try:
            self.driver.get(url)
            return self.driver.page_source
        except Exception as e:
            print(f"Error getting page content: {str(e)}")
            return ""

    async def scrape_article_content(self, url: str, content_selector: str) -> str:
        try:
            page_content = await self.get_page_content(url)
            if not page_content:
                return ""
            
            soup = BeautifulSoup(page_content, 'html.parser')
            content_element = soup.select_one(content_selector)
            return content_element.get_text().strip() if content_element else ""
            
        except Exception as e:
            print(f"Error scraping content: {str(e)}")
            return ""

    def get_existing_articles(self) -> dict:
        """이전에 스크랩된 기사들의 정보를 로드합니다."""
        existing_articles = {}
        
        try:
            # 최근 30일간의 CSV 파일들을 확인
            for i in range(30):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                for source in self.news_sources.keys():
                    filename = os.path.join(
                        self.data_dir,
                        f'{source.lower()}_news_{date}.csv'
                    )
                    if os.path.exists(filename):
                        df = pd.read_csv(filename)
                        for _, row in df.iterrows():
                            existing_articles[row['link']] = {
                                'first_seen': row.get('first_seen', row['date']),
                                'content': row.get('original_content', '')
                            }
        except Exception as e:
            print(f"Error loading existing articles: {str(e)}")
        
        return existing_articles

    def is_content_changed(self, old_content: str, new_content: str) -> bool:
        """컨텐츠가 실질적으로 변경되었는지 확인합니다."""
        def normalize_content(content: str) -> str:
            # 공백, 특수문자 등을 제거하고 정규화
            return ' '.join(content.strip().split())
        
        old_normalized = normalize_content(old_content or '')
        new_normalized = normalize_content(new_content or '')
        
        # 해시값을 비교하여 변경 여부 확인
        old_hash = hashlib.md5(old_normalized.encode()).hexdigest()
        new_hash = hashlib.md5(new_normalized.encode()).hexdigest()
        
        return old_hash != new_hash

    async def scrape_news(self, source_name: str, limit: int = 20) -> List[Article]:
        global scraping_status
        source_info = self.news_sources.get(source_name)
        
        if not source_info:
            raise HTTPException(status_code=404, detail=f"Source {source_name} not found")
        
        articles = []
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Get page content
                page_content = await self.get_page_content(source_info['url'])
                if not page_content:
                    print(f"Empty page content received for {source_name}, attempt {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    await asyncio.sleep(5)
                    continue
                
                # Parse the page
                soup = BeautifulSoup(page_content, 'html.parser')
                print(f"Scraping {source_name}...")
                
                # Get elements with safety checks
                title_elements = soup.select(source_info['title_selector'])[:limit]
                link_elements = soup.select(source_info['link_selector'])[:limit]
                
                if not title_elements or not link_elements:
                    print(f"No elements found for {source_name}, attempt {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    await asyncio.sleep(5)
                    continue
                
                # Process each article
                for title_elem, link_elem in zip(title_elements, link_elements):
                    try:
                        # Extract and validate title
                        article_text = title_elem.get_text().strip() if title_elem else ""
                        if not article_text:
                            continue
                        
                        # Extract and validate link
                        article_link = link_elem.get('href', "").strip() if link_elem else ""
                        if not article_link:
                            continue
                        
                        # Ensure proper URL format
                        if not article_link.startswith(('http://', 'https://')):
                            article_link = f"https://www.{source_name.lower()}.com{article_link}" if not article_link.startswith('/') else f"https://www.{source_name.lower()}.com{article_link}"
                        
                        # Create article object
                        article = Article(
                            source=source_name,
                            original_title=article_text,
                            korean_title=await self.translator.translate_text(article_text),
                            link=article_link,
                            date=current_date,
                            first_seen=current_date,
                            last_updated=current_date
                        )
                        
                        articles.append(article)
                        await asyncio.sleep(random.uniform(1, 2))
                        
                    except Exception as e:
                        print(f"Error processing individual article: {str(e)}")
                        traceback.print_exc()
                        continue
                
                # Save articles
                if articles:
                    try:
                        df = pd.DataFrame([article.dict() for article in articles])
                        csv_filename = os.path.join(
                            self.data_dir,
                            f'{source_name.lower()}_news_{datetime.now().strftime("%Y%m%d")}.csv'
                        )
                        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                    except Exception as e:
                        print(f"Error saving articles to CSV: {str(e)}")
                        traceback.print_exc()
                
                # Update scraping status
                scraping_status["sources_completed"].append(source_name)
                scraping_status["total_articles"] += len(articles)
                
                # Successfully scraped, break the retry loop
                break
                
            except Exception as e:
                print(f"Error scraping {source_name}: {str(e)}")
                traceback.print_exc()
                retry_count += 1
                await asyncio.sleep(5)
        
        return articles

@app.get("/")
async def root():
    return {"message": "Welcome to Marketing News API"}

@app.get("/sources")
async def get_sources():
    return {"sources": list(NewsSource)}

@app.get("/news/{source}", response_model=List[Article])
async def get_news(source: NewsSource, limit: Optional[int] = 20):
    scraper = MarketingNewsScraper()
    try:
        articles = await scraper.scrape_news(source.value, limit)
        return articles
    finally:
        scraper.cleanup()

@app.get("/status")
async def get_status():
    return scraping_status

@app.get("/scrape-all", response_model=List[Article])
async def scrape_all_sources(background_tasks: BackgroundTasks):
    global scraping_status
    
    if scraping_status["is_running"]:
        raise HTTPException(status_code=400, detail="Scraping is already in progress")
    
    scraping_status.update({
        "is_running": True,
        "status": "running",
        "sources_completed": [],
        "total_articles": 0
    })
    
    try:
        scraper = MarketingNewsScraper()
        all_articles = []
        
        for source in NewsSource:
            articles = await scraper.scrape_news(source.value)
            all_articles.extend(articles)
            await asyncio.sleep(random.uniform(2, 4))
        
        scraping_status.update({
            "status": "completed",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return all_articles
        
    except Exception as e:
        scraping_status["status"] = "failed"
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        scraping_status["is_running"] = False
        if 'scraper' in locals():
            scraper.cleanup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)