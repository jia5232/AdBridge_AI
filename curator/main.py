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
    original_content: Optional[str] = None
    korean_content: Optional[str] = None
    first_seen: str  # 처음 스크랩된 날짜
    last_updated: str  # 마지막으로 업데이트된 날짜

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
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        chrome_options.page_load_strategy = 'eager'
        
        # ChromeDriverManager 설정 수정
        driver_manager = ChromeDriverManager()
        driver_path = driver_manager.install()
        
        service = Service(
            executable_path=driver_path,
            # Mac M1/M2를 위한 추가 설정
            log_path=os.devnull
        )
        
        try:
            self.driver = webdriver.Chrome(
                service=service, 
                options=chrome_options
            )
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(30)
        except Exception as e:
            print(f"WebDriver 초기화 오류: {str(e)}")
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
        
        # 기존 기사 데이터 로드
        existing_articles = self.get_existing_articles()
        articles = []
        new_or_updated_articles = []
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                page_content = await self.get_page_content(source_info['url'])
                if not page_content:
                    retry_count += 1
                    continue
                
                soup = BeautifulSoup(page_content, 'html.parser')
                
                title_elements = soup.select(source_info['title_selector'])[:limit]
                link_elements = soup.select(source_info['link_selector'])[:limit]
                
                for title_elem, link_elem in zip(title_elements, link_elements):
                    article_text = title_elem.get_text().strip()
                    article_link = link_elem.get('href')
                    
                    if not article_link.startswith('http'):
                        article_link = f"https://{article_link.lstrip('/')}"
                    
                    # 새 컨텐츠 스크랩
                    content = await self.scrape_article_content(
                        article_link,
                        source_info['content_selector']
                    )
                    
                    existing_data = existing_articles.get(article_link)
                    
                    # 새 기사이거나 컨텐츠가 변경된 경우에만 처리
                    if not existing_data or self.is_content_changed(existing_data['content'], content):
                        translated_title = await self.translator.translate_text(article_text)
                        translated_content = await self.translator.translate_text(content) if content else None
                        
                        article = Article(
                            source=source_name,
                            original_title=article_text,
                            korean_title=translated_title,
                            link=article_link,
                            date=current_date,
                            original_content=content,
                            korean_content=translated_content,
                            first_seen=existing_data['first_seen'] if existing_data else current_date,
                            last_updated=current_date
                        )
                        
                        new_or_updated_articles.append(article)
                    
                    articles.append(article)
                    await asyncio.sleep(random.uniform(1, 2))
                
                # 새로운 또는 업데이트된 기사가 있는 경우에만 CSV 저장
                if new_or_updated_articles:
                    df = pd.DataFrame([article.dict() for article in new_or_updated_articles])
                    csv_filename = os.path.join(
                        self.data_dir,
                        f'{source_name.lower()}_news_{datetime.now().strftime("%Y%m%d")}.csv'
                    )
                    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                
                scraping_status["sources_completed"].append(source_name)
                scraping_status["total_articles"] += len(new_or_updated_articles)
                break
                
            except TimeoutException:
                retry_count += 1
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Error scraping {source_name}: {str(e)}")
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