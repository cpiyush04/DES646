import json
from flask import Flask, render_template, request, Response, jsonify
from typing import List, Dict, Generator
from agno.agent import Agent
import time
from agno.models.google import Gemini
from agno.os import AgentOS
from dotenv import load_dotenv
import os
from agno.team import Team
import requests
from typing import List, Dict
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from tavily import TavilyClient
from urllib.parse import urlparse
import json
from agno.workflow import Parallel, Step, Workflow, StepOutput
import re
from datetime import datetime

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY1 = os.getenv("GOOGLE_API_KEY1")
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Model & Client Initialization ---
model1 = Gemini(id="gemini-2.5-flash", api_key=GOOGLE_API_KEY1)
model2 = Gemini(id="gemini-2.5-flash", api_key=GOOGLE_API_KEY2)

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables.")
client = TavilyClient(api_key=TAVILY_API_KEY)

app = Flask(__name__)

#

# --- (Example) Pasted Agent & Tool Functions ---

import json
import re
from typing import List

def extract_content_from_urls(urls: str):
    """
    Extracts content from a list of URLs using TavilyClient.
    Skips any URL that fails extraction (e.g., blocked pages, timeouts, etc.).

    Args:
        urls (str): A JSON string containing a list of URLs.

    Returns:
        list[dict]: A list of extracted articles (url, title, raw_content).
    """
    # 1️⃣ Clean and parse JSON list of URLs
    cleaned = re.sub(r'```|json', '', urls).strip()
    try:
        parsed = json.loads(cleaned)
# support both {"urls": [...]} and a raw list
        url_list = parsed["urls"] if isinstance(parsed, dict) and "urls" in parsed else parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in URLs input: {e}")

    extracted_content = []

    # 2️⃣ Try extracting each URL individually (so we skip failed ones)
    for url in url_list:
        try:
            print("extrcating", url)
            response = client.extract(urls=[url])  # extract one by one
            results = response.get('results', [])

            if not results:
                print(f"⚠️ No results returned for: {url}")
                continue

            item = results[0]
            content = item.get('raw_content', '')
            title = item.get('title', '')

            if not content:
                print(f"⚠️ Empty content for: {url}")
                continue

            extracted_content.append({
                "url": url,
                "title": title,
                "raw_content": content
            })

        except Exception as e:
            print(f"❌ Skipping {url} due to error: {e}")
            continue

    # 3️⃣ Return as list (you can wrap it in StepOutput if needed)
    return extracted_content


def search_over_newsapi(query: str) -> list[dict]:
    """
    Searches NewsAPI for the top 15 recent articles on a given topic.

    Args:
        query (str): The search query or topic.

    Returns:
        list[dict]: A list of dictionaries with 'title', 'url', and 'source' for each article,
                    or an error message if something fails.
    """
    api_key = NEWS_API_KEY
    base_url = "https://newsapi.org/v2/everything"

    if not api_key:
        return [{"error": "News API key is not set."}]

    params = {
        'q': query,
        'pageSize': 15,
        'apiKey': api_key,
        'language': 'en'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        print(response.json())
        return [
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source", {}).get("name")
            }
            for article in data.get("articles", [])
        ]
    except Exception as e:
        return [{"error": f"News API request failed: {e}"}]

# Create the agent
news_search_agent = Agent(
    name="Search & Deduplication Specialist",
    role="Find news articles from multiple sources and remove exact URL duplicates",
    model=model1,
    tools=[
        GoogleSearchTools(all=True),
        DuckDuckGoTools(all=True),
        search_over_newsapi
    ],
    instructions=[
        "You are a search specialist responsible for gathering news articles from multiple sources",
        
        "STEP 1: PARALLEL SEARCH",
        "- First correct the user's query for any typos or errors as search tools are sensitive to spelling or punctuations mistakes",
        " For example, user wrote 'H1B visa'but correct query is 'H-1B visa'",
        "- Search all three sources simultaneously: Google, DuckDuckGo, and NewsAPI",
        "- Request 15 articles from each source (total ~50 articles)",
        "- Use the exact same search query for all three sources",
        "- Each article MUST include: url, title, thumbnail (if available) and source name",
        
        "STEP 2: FILTERING",
        "- After collecting all articles, select ABOUT 6 diverse articles covering various different aspects and sentiments from different sources.",
        "- Prepare a list 'urls' of urls from the selected articles",
        
        "RETURN THE LIST OF URLS IN JSON FORMAT",
        "**DO NOT RETURN ANY OTHER TEXT. JUST RETURN VALID JSON REPORT NOTHING ELSE IS ALLOWED.**"
        
    ],
    markdown= False,
    debug_mode = True,
)

news_filter_agent = Agent(
    name="News Filtering Expert",
    id="News Filtering Expert",
    role="Refine and clean scraped article text to retain only the meaningful news content.",
    # model=Gemini(id="gemini-2.0-flash"),
    model=model2,
    instructions=[
        "You are a news filtering expert.",
        "Your task is to clean the article text received",
        "",
        "CLEANING RULES:",
        "- Do NOT summarize, shorten, or paraphrase the article.",
        "- Remove any ads, subscription prompts, or promotional lines like 'Advertisement', 'Subscribe Now', 'Trending', or 'Explore More'.",
        "- Remove navigation headers, unrelated article titles, and social media references.",
        "- Remove irrelevant characters like \n, \t, \r, and unnecessary symbols present in text.",
        "- Remove image markdowns like ![](url) or [text](url).",
        "- Keep the full, original article content — only remove the noise.",
        "- Preserve paragraph breaks and quotations as they are.",
        "",
        "OUTPUT FORMAT:",
        "Return a clean JSON list of articles where each article includes:",
        "**DO NOT RETURN ANY OTHER TEXT. JUST RETURN VALID JSON REPORT NOTHING ELSE IS ALLOWED.**"
        "{ 'url': <url>, 'title': <title>, 'content': <cleaned full text> }",
    ],
    markdown=False,
    debug_mode=True,
)

news_coverage_agent = Agent(
    name="News Coverage Investigator",
    role="Analyze the entire corpus of articles to find the core factual claim(s) and coverage gaps",
    model=model1,
    instructions=[
        "You are a cross-article coverage analyst. Your job is to analyze ALL provided articles together.",
        "You will receive a list of articles; each article includes: url, title and content.",
        "EXECUTE THESE STEPS IN ORDER:",
        "STEP 1: READ ALL ARTICLES",
        "- Ensure you process every article in the input. At the start, assert and report the total number of articles received.",

        "STEP 2: SYNTHESIZE CORPUS-LEVEL CORE IDEA",
        "- Using all articles, explain the single core factual claim that the majority of articles support in paragraph.",
        "- Compute `agreement_pct` = percentage of articles that support that core claim (by checking claim presence).",
        "- List `supporting_articles` (urls) that support the claim and `contradicting_articles` (urls) if any.",
        "STEP 3: IDENTIFY COVERAGE GAPS",
        "- Identify deviating themes (coverage gaps) — distinct factual narratives or emphases present in subsets of articles.",
        "- For each gap, provide: gap_id, description about deviation from core news, pct_of_articles, and article urls.",
        "FINAL OUTPUT:",
        "- Return a complete report in JSON that includes:",
        "  * input_count and assertion that all articles were processed",
        "  * core_idea + agreement_pct + supporting_article list",
        "  * coverage_gaps with descriptions and percentages",
        
        "**DO NOT RETURN ANY OTHER TEXT. JUST RETURN VALID JSON REPORT NOTHING ELSE IS ALLOWED.**"
        "JSON FORMAT:"
        """
        {
        "agent": "news_coverage_agent",
        "data": {
            "input_count": 13,
            "processing_assertion": "All 13 articles were processed.",
            "core_idea": {
                "claim": "Multiple explosions occurred at a high school mosque in Jakarta, Indonesia, during Friday prayers, injuring dozens of people, and a 17-year-old student has been identified as the suspected perpetrator.",
                "agreement_pct": 100.0,
                "supporting_articles": [
                    "https://www.cnn.com/2025/11/07/asia/indonesia-mosque-high-school-jakarta-explosion-latam-intl",
                    "https://www.cbsnews.com/news/jakarta-indonesia-explosion-high-school-mosque-friday-prayers/",
                    "https://www.abc.net.au/news/2025-11-08/police-investigate-suspect-mosque-explosion-radicalised/105988228",
                    ....
                ],
                "contradicting_articles": []
            },
            "coverage_gaps": [
                {
                    "gap_id": "explicit_extremist_ideology_probe",
                    "description": "Some articles explicitly detail the extremist inscriptions found on the toy weapon ('14 words,' 'Brenton Tarrant: Welcome to hell'), linking them to white supremacist ideology and the Christchurch mosque shooter. These articles often focus on police investigations into whether the suspect was radicalized and exploring ideological motives, sometimes also mentioning the finding of 'potentially explosive powder' at the suspect's home.",
                    "pct_of_articles": 38.46,
                    "article_urls": [
                        "https://www.cnn.com/2025/11/07/asia/indonesia-mosque-high-school-jakarta-explosion-latam-intl",
                        "https://www.abc.net.au/news/2025-11-08/police-investigate-suspect-mosque-explosion-radicalised/105988228",
                        "https://apnews.com/article/indonesia-explosions-mosque-high-school-students-366d88d30e28bac9a5b2ac3b43de9b4f",
                        
                    ]
                },
                {
                    "gap_id": "updated_casualty_and_investigation_progress",
                    "description": "A subset of articles provides updated and significantly higher casualty figures (e.g., 'nearly 100' or 96 injured) compared to the initial reports of 50-55. These articles also tend to include the latest investigative findings, such as the discovery of 'potentially explosive powder' at the suspect's home, and updates on the suspect's medical condition and cooperation with authorities.",
                    "pct_of_articles": 23.08,
                    "article_urls": [
                        "https://www.abc.net.au/news/2025-11-08/police-investigate-suspect-mosque-explosion-radicalised/105988228",
                        "https://www.yahoo.com/news/articles/indonesia-probes-student-nearly-100-092038973.html",
                        "https://www.channelnewsasia.com/asia/indonesia-mosque-explosion-police-explosive-5455076"
                    ]
                },
            ]
        }
    },
        """

    ],
    markdown=False,
    debug_mode=True,
)

credibility_agent = Agent(
    name="Credibility Assessor",
    role="Evaluate credibility for every article and produce per-article scores and a top-10 ranking",
    # model=Gemini(id="gemini-2.5-flash-lite"),
    model=model2,
    instructions=[
        "You are a news credibility assessor. Your job is to evaluate the credibility of EVERY provided article.",
        "You will receive a list of articles with metadata and full text.",
        "EXECUTE THESE STEPS IN ORDER:",
        "STEP 1: READ FULL ARTICLE TEXT FOR EACH ARTICLE",
        "- Process all articles sequentially (or in batches) and ensure the total processed equals the input count.",
        "STEP 2: SCORE EACH ARTICLE (0-100) USING THE FOLLOWING RUBRIC",
        "- Source reputation (weight 0.40)",
        "- Presence of citations/links to primary sources (weight 0.15)",
        "- Named primary sources / number of direct quotes from identified sources (weight 0.15)",
        "- Author byline & author credibility signals (weight 0.05)",
        "- Cross-article corroboration (weight 0.25)",
        "- For each component, normalize scores to a 0–100 final credibility_score.",
        "STEP 3: AGGREGATE & RANK",
        "- After scoring all articles, produce a list sorted by credibility_score (include id, title, source, score).",
        "FINAL OUTPUT:",
        "- Return a JSON report including: id, title, source, credibility_score",

        "**DO NOT RETURN ANY OTHER TEXT. JUST RETURN VALID JSON REPORT NOTHING ELSE IS ALLOWED.**"
                "JSON Format:"
        """
          {
        "agent": "credibility_agent",
        "data": [
            {
                "id": "apnews.com/article/indonesia-explosions-mosque-high-school-students-366d88d30e28bac9a5b2ac3b43de9b4f",
                "title": "Explosions at a high school mosque in Indonesia injure 55 - AP News",
                "source": "AP News",
                "credibility_score": 96.25
            },
            {
                "id": "https://www.cnn.com/2025/11/07/asia/indonesia-mosque-high-school-jakarta-explosion-latam-intl",
                "title": "Multiple explosions shake mosque at Indonesian high school ... -...",
                "source": "CNN",
                "credibility_score": 93.75
            },
        ]
    },
        """
    ],
    markdown=False,
    debug_mode=True,
)

sentiment_agent = Agent(
    name="Sentiment & Tone Classifier",
    role="Classify sentiment and tone for every article and cluster the corpus by sentiment labels",
    # model=Gemini(id="gemini-2.0-flash-lite"),
    model=model1,
    instructions=[
        "You are a news sentiment and tone classifier. You must analyze EVERY article provided.",
        "Input: list of articles (url, title, full text).",
        "EXECUTE THESE STEPS IN ORDER:",
        "STEP 1: READ FULL ARTICLE TEXT",
        "- Confirm total articles processed equals input count and report it.",
        "STEP 2: PER-ARTICLE SENTIMENT & TONE",
        "- For each article, determine:",
        "  * sentiment_label: Positive / Neutral / Negative",
        "  * sentiment_score: float between -1.0 (very negative) and +1.0 (very positive)",
        "  * confidence: 0.0–1.0",
        "  * tone: one of [Factual, Opinion, Analysis, Editorial]",
        "- Provide 1–3 short evidence excerpts (phrases or sentences) that justify the label.",
        "STEP 3: CORPUS-LEVEL AGGREGATION",
        "- Aggregate counts and percentages for each sentiment label across the entire corpus.",
        "- Provide clusters: for each label, list representative article urls (up to 5 examples).",
        "FINAL OUTPUT:",
        "- Return a JSON report that shows per-article sentiment rows and a summary section with distribution percentages and example headlines.",

        "OUTPUT SCHEMA:",
        "{",
        '  "total_articles_processed": integer,',
        '  "articles": [',
        '      {',
        '          "url": string,',
        '          "title": string,',
        '          "sentiment_label": string,',
        '          "sentiment_score": float,',
        '          "confidence": float,',
        '          "tone": string,',
        '          "evidence": [',
        '              string',
        '          ]',
        '      }',
        '  ],',
        '  "summary": {',
        '      "sentiment_distribution": {',
        '          "Positive": {',
        '              "count": integer,',
        '              "percentage": string',
        '          },',
        '          "Neutral": {',
        '              "count": integer,',
        '              "percentage": string',
        '          },',
        '          "Negative": {',
        '              "count": integer,',
        '              "percentage": string',
        '          }',
        '      },',
        '      "sentiment_clusters": {',
        '          "Positive": [ string ],',
        '          "Neutral": [ string ],',
        '          "Negative": [ string ]',
        '      }',
        '  }',
        "}"

        "**DO NOT RETURN ANY OTHER TEXT. JUST RETURN VALID JSON REPORT NOTHING ELSE IS ALLOWED.**"
    ],
    markdown=False,
    debug_mode=True,
)

share_of_voice_agent = Agent(
    name="Share of Voice Extractor",
    role="Detect and quantify stakeholder mentions across the full corpus and compute share-of-voice percentages",
    model=model2,
    instructions = [
    "You are a stakeholder extraction and share-of-voice analyst. Process ALL articles provided.",
    "Input: list of articles (url, title, full text).",
    "EXECUTE THESE STEPS IN ORDER:",
    
    "STEP 1: READ FULL ARTICLE TEXT FOR EACH ARTICLE",
    "- Ensure you process every article; report input_count == processed_count.",
    "- Extract all explicit stakeholder mentions: people, organizations, collectives, and generic stakeholder nouns (e.g., 'farmers', 'students', 'residents').",
    "- For each mention include: canonical_name, percentage(that mention appeared in all articles) and example quote (if quoted).",
    
    "STEP 2: MAP TO CANONICAL BUCKETS",
    "- Map extracted entities into canonical stakeholder buckets: Politicians, Farmers, Students, Businesses, NGOs, Journalists, Law Enforcement, Others.",
    "- If mapping is ambiguous, include internal resolution but do NOT output mapping_confidence or uncertainty scores.",
    
    "STEP 3: AGGREGATE SHARE-OF-VOICE",
    "- Across the corpus compute total_mentions and quoted_mentions per bucket and convert to percentages summing to 100%.",
    "- For each bucket include: percentage, and examples (up to 5 representative stakeholder mentions).",
    "- Each example must include canonical_name, prcentage, example_quote, and article_id (URL).",
    
    "STEP 4: COMPUTE TOP ENTITIES",
    "- Identify the top 10 most-mentioned canonical entities across all articles.",
    "- For each entity include: canonical_name, percentage mentions and example_article_ids (list of up to 7 URLs).",
    "- Sort by total_mentions descending.",
    
    "STEP 5: VALIDATE COUNTS",
    "- Include 'input_count' (total articles received) and 'processed_count' (articles successfully processed).",
    "- Ensure processed_count == input_count unless some articles are missing or malformed.",
    
    "FINAL OUTPUT:",
    "- Return one single JSON object and nothing else (no markdown, no commentary).",
    "- The final JSON must include only FOUR top-level keys, in this order:",
    "   1. input_count",
    "   2. processed_count",
    "   3. stakeholder_mentions (the Share of Voice summary per bucket)",
    "   4. top_10_entities (the overall top entities list)",
    "- No other fields such as full_entity_extraction, mapping_confidence, or debug info are allowed.",
    "**DO NOT RETURN ANY OTHER TEXT. JUST RETURN VALID JSON REPORT NOTHING ELSE IS ALLOWED.**"
    
    "OUTPUT SCHEMA:",
    "{",
    '  "input_count": integer,',
    '  "processed_count": integer,',
    '  "stakeholder_mentions": {',
    '      "Politicians": {',
    '          "percentage": float,',
    '          "examples": [',
    '              {',
    '                  "canonical_name": string,',
    '                  "example_quote": string,',
    '                  "article_id": string',
    '              }',
    '          ]',
    '      },',
    '      "Farmers": {...},',
    '      "Students": {...},',
    '      "Businesses": {...},',
    '      "NGOs": {...},',
    '      "Journalists": {...},',
    '      "Law Enforcement": {...},',
    '      "Others": {...}',
    '  },',
    '  "top_10_entities": [',
    '      {',
    '          "canonical_name": string,',
    '           "percentage": float,',
    '          "example_article_ids": [string, ...]',
    '      }',
    '  ]',
    "}",
    
    "VALIDATION RULES:",
    "- Percentages across stakeholder buckets must sum to 100.00 (±0.1 tolerance).",
    "- All integers must be ≥ 0.",
    "- top_10_entities length ≤ 10.",
    "- Output must be valid JSON with no markdown or extra text before/after.",
    
    "EXAMPLE OUTPUT:",
    "{",
    '  "input_count": 12,',
    '  "processed_count": 12,',
    '  "stakeholder_mentions": {',
    '      "Politicians": {',
    '          "percentage": 42.31,',
    '          "examples": [',
    '              {',
    '                  "canonical_name": "Donald Trump",',
    '                  "example_quote": "US President Donald Trump on Thursday expressed that he is likely to visit India next year while praising Prime Minister Narendra Modi and calling him a ‘great man’.",',
    '                  "article_id": "https://www.outlookbusiness.com/economy-and-policy/us-president-trump-likely-to-visit-india-next-year-calls-pm-modi-his-friend-and-a-great-man"',
    '              }',
    '          ]',
    '      },',
    '      "Farmers": { ""percentage": 0.00, "examples": [] }',
    '  },',
    '  "top_10_entities": [',
    '      {',
    '          "canonical_name": "Donald Trump",',
    '          "percentage": 33%,',
    '          "example_article_ids": [',
    '              "https://www.outlookbusiness.com/...",',
    '              "https://www.nytimes.com/..."',
    '          ]',
    '      }',
    '  ]',
    "}"
],
    markdown=False,
    debug_mode=True,
)

def safe_json_loads(data, agent_name):
        try:
            if data.strip().startswith("```json") or data.strip().startswith("json```"):
            # Remove any code fences or 'json' markers
                cleaned = re.sub(r'```|json', '', data, flags=re.IGNORECASE).strip()
                return json.loads(cleaned)
            return json.loads(data)
        except Exception as e:
            return {"agent": agent_name, "error": str(e), "raw_output": data}

    
def process_pipeline_results(all_agent_data: dict):
    """
    Formats the final dictionary of all agent results for the frontend.
    This is the same as your old 'process_pipeline_results' but takes a dict.
    """
    dashboard_data = {
        "core_story": {"description": "N/A", "agreement_pct": 0},
        "coverage_gaps": [],
        "credibility_scores": [],
        "sentiment_summary": {"distribution": {}},
        "top_entities": []
    }
    
    try:
            
        news_coverage_data = all_agent_data.get('news_coverage_agent', {})
        credibility_data = all_agent_data.get('credibility_agent', []).get('data',[])
        sentiment_data = all_agent_data.get('sentiment_agent', {})
        share_of_voice_data = all_agent_data.get('share_of_voice_agent', {})
        
        # ---
        # ✅ FIX 1: Handle Core Story being an object
        # ---
        core_idea_data = news_coverage_data.get('data').get('core_idea')
        # print(core_idea_data)
        core_desc = "No core story found."
        core_pct = 0.0  # Default percentage
        
        if isinstance(core_idea_data, dict):
            # If it's a dict, get claim AND percentage from inside it
            core_desc = core_idea_data.get('claim', core_idea_data.get('description', str(core_idea_data)))
            # <-- Get pct from *inside* the core_idea_data object
            core_pct = core_idea_data.get('agreement_pct', 0.0) 
        elif isinstance(core_idea_data, str) and core_idea_data:
            # If it's just a string, use it
            core_desc = core_idea_data
            # And try to get the percentage from the top level (fallback)
            core_pct = news_coverage_data.get('agreement_pct', 0.0) 
        
        dashboard_data["core_story"] = {
            "description": core_desc,
            "agreement_pct": core_pct 
        }
        
        # 2. Credibility (Sorted)
        # print(credibility_data)
        if isinstance(credibility_data, list):
            dashboard_data["credibility_scores"] = sorted(
                credibility_data, 
                key=lambda x: x.get('credibility_score', 0), 
                reverse=True
            )
        else:
            dashboard_data["credibility_scores"] = sorted(
                credibility_data.get('results', []) if isinstance(credibility_data, dict) else [],
                key=lambda x: x.get('credibility_score', 0),
                reverse=True
            )
        
        # 3. Coverage Gaps (Sorted)
        dashboard_data["coverage_gaps"] = sorted(
            news_coverage_data.get('data').get('coverage_gaps', []), 
            key=lambda x: x.get('pct_of_articles', 0), 
            reverse=True
        )

        # ---
        # ✅ FIX 2: Parse Sentiment Percentage string to float
        # ---
        corpus_summary = sentiment_data.get('summary', sentiment_data if isinstance(sentiment_data, dict) else {})
        dist = corpus_summary.get('sentiment_distribution', {})
        
        for key in ["Negative", "Neutral", "Positive"]:
            # Ensure the entry for the key is a dictionary
            if not isinstance(dist.get(key), dict):
                dist[key] = {"percentage": 0, "count": 0}

            # Get the percentage, which might be a string like "25.5%" or a number
            raw_pct = dist[key].get("percentage", 0)
            
            clean_pct = 0.0
            try:
                # Convert to string, strip '%' and whitespace, then convert to float
                clean_pct = float(str(raw_pct).strip().replace('%', ''))
            except (ValueError, TypeError):
                clean_pct = 0.0 # Default to 0 if conversion fails
                
            dist[key]["percentage"] = clean_pct # Store the clean float
            
            # Ensure count exists
            if "count" not in dist[key]:
                 dist[key]["count"] = 0
                 
        dashboard_data["sentiment_summary"]["distribution"] = dist
        # --- END FIX 2 ---

        # Extract Sentiment Clusters
        clusters = corpus_summary.get('sentiment_clusters', {})
        dashboard_data["sentiment_summary"]["clusters"] = {
            "Positive": clusters.get("Positive", []),
            "Neutral": clusters.get("Neutral", []),
            "Negative": clusters.get("Negative", [])
        }

        # 5. Top Entities
        dashboard_data["top_entities"] = share_of_voice_data.get('top_10_entities', [])
        # print(dashboard_data)

        return dashboard_data
        
    except Exception as e:
        print(f"Error processing pipeline results: {e}")
        return {"error": f"Error processing results: {e}"}

def stream_news_pipeline(query: str) -> Generator[str, None, None]:
    """
    Executes the full news analysis pipeline, yielding status updates
    as JSON strings for Server-Sent Events (SSE).
    """
    
    def stream(step: int, status: str, message: str, data: dict = None):
        """Helper to format the SSE message."""
        event_data = {"step": step, "status": status, "message": message, "data": data}
        return f"data: {json.dumps(event_data)}\n\n"

    all_results = {}
    
    try:
        # Step 1: Run search agent
        yield stream(1, "running", "Searching for articles...")
        search_result = news_search_agent.run(query).content
        
        # # Step 2: Extract content from URLs
        yield stream(2, "running", "Found articles, extracting content...")
        extracted_articles = extract_content_from_urls(search_result)
        if not extracted_articles:
             yield stream(0, "error", "No articles could be extracted. Please try a different query.")
             return

        # # Step 3: Filter relevant articles
        yield stream(3, "running", "Filtering and cleaning articles...")
        filtered_articles_raw = news_filter_agent.run(extracted_articles).content
        filtered_articles = safe_json_loads(filtered_articles_raw, "news_filter_agent")
        if "error" in filtered_articles:
            yield stream(0, "error", f"Failed to filter articles: {filtered_articles['error']}")
            return

        # # Step 4: Run parallel analysis agents
        # # We run them sequentially here, but stream after each one.
        
        yield stream(4, "running", "Analyzing news coverage and gaps...")
        news_coverage_raw = news_coverage_agent.run(filtered_articles).content
        all_results["news_coverage_agent"] = safe_json_loads(news_coverage_raw, "news_coverage_agent")


        yield stream(5, "running", "Assessing article credibility...")
        credibility_score_raw = credibility_agent.run(filtered_articles).content
        all_results["credibility_agent"] = safe_json_loads(credibility_score_raw, "credibility_agent")

        yield stream(6, "running", "Classifying sentiment and tone...")
        sentiment_res_raw = sentiment_agent.run(filtered_articles).content
        all_results["sentiment_agent"] = safe_json_loads(sentiment_res_raw, "sentiment_agent")

        yield stream(7, "running", "Extracting share of voice...")
        voice_share_raw = share_of_voice_agent.run(filtered_articles).content
        all_results["share_of_voice_agent"] = safe_json_loads(voice_share_raw, "share_of_voice_agent")

        # # Step 5: Process and send final payload
        yield stream(8, "processing", "All analysis complete. Generating final dashboard...")

        final_dashboard_data = process_pipeline_results(all_results)
        
        if "error" in final_dashboard_data:
             yield stream(0, "error", final_dashboard_data["error"])
        else:
            yield stream(9, "complete", "Done.", data=final_dashboard_data)



    except Exception as e:
        print(f"Pipeline failed: {e}")
        yield stream(0, "error", f"An unexpected error occurred: {str(e)}")


# --- UPDATED: Flask Routes ---

# <-- 2. CORRECTED This route
@app.route('/', methods=['GET'])
def dashboard():
    """Renders the main dashboard page."""
    # The page is now just a static shell, JavaScript handles everything.
    return render_template('index.html')

@app.route('/get-top-news')
def get_top_news():
    """
    Fetches top headlines from NewsAPI to populate the sidebar.
    """
    if not NEWS_API_KEY:
        print("Error: NEWS_API_KEY not found.")
        return jsonify({"error": "News API key not configured"}), 500
    
    api_url = "https://newsapi.org/v2/top-headlines"
    params = {
        'country': 'us',
        'apiKey': NEWS_API_KEY,
        'pageSize': 5 # We only need the top 5
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status() # Raise an error for bad responses
        data = response.json()
        
        # Check NewsAPI status
        if data.get("status") != "ok":
            print(f"NewsAPI error: {data.get('message')}")
            return jsonify({"error": "NewsAPI returned an error"}), 500
            
        return jsonify(data.get("articles", []))
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching top news: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/stream-analysis')
def stream_analysis():
    """
    The new streaming endpoint. JavaScript connects to this.
    """
    query = request.args.get('query')
    if not query:
        return Response("Error: No query provided.", status=400)
    
    # Return a streaming response
    return Response(
        stream_news_pipeline(query),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no' # Important for Gunicorn
        }
    )

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    print("Starting Flask app in streaming mode...")
    print("Visit http://127.0.0.1:5000 in your browser.")
    # Note: app.run(debug=True) can sometimes interfere with SSE.
    # For production, Gunicorn is used (defined in Dockerfile).
    app.run(debug=True)