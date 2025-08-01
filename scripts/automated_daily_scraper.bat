@echo off
REM Daily Reddit Scraper + Preprocessing- Batch Script for Windows Task Scheduler

REM Project directory
cd "C:\Users\necke\Desktop\reddit depression\Reddit_Data_Project\Reddit_Data_Project"

REM Activate conda environment
call C:\Users\necke\anaconda3\Scripts\activate.bat base

REM Run the scraper
python scripts\reddit_scraper.py

REM Run the preprocessing
python scripts\preprocess_all.py

REM Log the completion with timestamp
echo %date% %time% - Daily scraping and preprocessing completed >> logs\daily_scraper.log

echo All tasks completed
pause