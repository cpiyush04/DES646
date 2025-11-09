# 📰 News Analysis Engine

**News Analysis Engine** is a web-based AI-powered platform that analyzes news articles from multiple sources in real-time.  
Built with the **agno framework** and powered by **Google Gemini**, it provides insights into media narratives, sentiment, credibility, and key stakeholders around any given topic.

---

## 🚀 Overview

A user enters a search query (e.g., *"Global AI Regulation"*), and the system automatically:
1. Fetches relevant news articles from multiple sources.  
2. Scrapes and cleans the content.  
3. Runs a **multi-agent AI analysis pipeline**.  
4. Streams real-time progress and insights to a dynamic dashboard.

The final dashboard includes:
- Core story synthesis  
- Sentiment distribution  
- Credibility assessment  
- Coverage gaps and nuance detection  
- Stakeholder analysis  

---

## ✨ Key Features

### 🔍 Multi-Source Search
Fetches and aggregates news articles using:
- **Google Search**
- **DuckDuckGo**
- **NewsAPI**

### 🧠 AI-Powered Content Extraction
Uses the **Tavily API** to extract clean, relevant content from each article URL.

### 🤖 Multi-Agent Analysis Pipeline
Built with the **agno** framework and **Google Gemini (gemini-2.5-flash)**, this pipeline analyzes each article for:
- **Core Story:** Synthesizes the main factual claim and computes agreement percentage.  
- **Overall Sentiment:** Displays positive, neutral, and negative proportions in a Chart.js doughnut chart.  
- **Key Nuances & Gaps:** Identifies missing or deviating narratives.  
- **Share of Voice:** Detects and ranks the most mentioned people and organizations.  
- **Credibility:** Scores articles based on a defined rubric, displayed in a sortable list.

### ⚡ Real-Time Streaming
- Flask backend uses **Server-Sent Events (SSE)** for real-time updates.  
- The frontend dynamically updates analysis progress and logs.

### 📊 Dynamic Dashboard
A modern single-page UI built with:
- **TailwindCSS**  
- **Chart.js**  
- **Vanilla JavaScript**

---

## 🧰 Tech Stack

| Layer | Tools / Frameworks |
|-------|--------------------|
| **Backend** | Python, Flask |
| **AI Framework** | agno (Agent Framework) |
| **AI Models** | Google Gemini (gemini-2.5-flash) |
| **Data Tools** | tavily-python, requests |
| **Search APIs** | googlesearch-python, ddgs (DuckDuckGo), NewsAPI |
| **Frontend** | HTML, TailwindCSS, Chart.js, JavaScript |

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

git clone <repository-url>
cd <repository-directory>

### 2. Create and Activate a Virtual Environment

python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Create Environment Variables

Create a .env file in the root directory and add the following keys:

GOOGLE_API_KEY1=YOUR_GOOGLE_API_KEY
GOOGLE_API_KEY2=YOUR_SECOND_GOOGLE_API_KEY
NEWS_API_KEY=YOUR_NEWSAPI_KEY
TAVILY_API_KEY=YOUR_TAVILY_API_KEY

Start the Flask development server:

python app.py

Then open your browser and visit:

http://127.0.0.1:5000


Enter a topic (e.g., Global AI Regulation) and click Analyze to start the pipeline.

🧱 Project File Structure
.
├── app.py                # Main Flask app, routes, and AI pipeline
├── test.py               # Standalone script to test the analysis pipeline
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
├── templates/
│   └── index.html        # Frontend: HTML, TailwindCSS, and JS
└── .gitignore            # Git ignore rules

🌐 API Endpoints
GET /

### Description:
Serves the main index.html dashboard page.

GET /get-top-news

Description:
Fetches the top 5 US headlines from NewsAPI to populate the sidebar.

Response Example:

{
  "status": "ok",
  "articles": [
    { "title": "Headline 1", "url": "https://..." },
    { "title": "Headline 2", "url": "https://..." }
  ]
}

GET /stream-analysis?query=<topic>

### Description:
Main analysis endpoint.
Takes a query parameter (e.g., /stream-analysis?query=Tesla).

Returns:
A text/event-stream (SSE) that streams real-time progress updates and final JSON output.

Example Streamed Event:

{
  "stage": "sentiment_analysis",
  "progress": 70,
  "data": {
    "positive": 45.2,
    "neutral": 35.6,
    "negative": 19.2
  }
}

## 🧩 Example Use Case

User searches “Electric Vehicle Market Outlook 2025.”

### The system:

Fetches 50+ recent articles.

Cleans and analyzes their content.

Streams a summary dashboard including:

📰 Core story: “EV adoption rising globally”

📈 Sentiment chart

🔑 Key players: Tesla, BYD, Toyota

🧮 Credibility scoring

## 🛠️ Future Enhancements

🌍 Support for multilingual article analysis

🧠 Integration with additional AI models (OpenAI, Anthropic, etc.)

📄 Export analysis reports to PDF/CSV

🔗 Advanced entity relationship graphs
