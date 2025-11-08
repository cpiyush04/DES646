from agno.agent import Agent
import time
from agno.models.google import Gemini
from agno.os import AgentOS
from dotenv import load_dotenv
import os
from agno.team import Team
import requests
from typing import List, Dict
from agno.tools.jina import JinaReaderTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from tavily import TavilyClient
from urllib.parse import urlparse
import json
from agno.workflow import Parallel, Step, Workflow, StepOutput
import re
from datetime import datetime
load_dotenv()
GOOGLE_API_KEY1 = os.getenv("GOOGLE_API_KEY1")
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

model1=Gemini(id="gemini-2.5-flash", api_key=GOOGLE_API_KEY1)
model2=Gemini(id="gemini-2.5-flash", api_key=GOOGLE_API_KEY2)

client = TavilyClient("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")

def extract_content_from_urls(urls: str):
    """
    Extracts content from a list of URLs using TavilyClient.

    Args:
        urls (str): A JSON string containing a list of URLs.

    Returns:
        StepOutput: Object containing the extracted articles as a JSON string
                    under .content, suitable for downstream agent input.
    """
    # 1️⃣ Remove any Markdown code fences or the word 'json'
    cleaned = re.sub(r'```|json', '', urls).strip()

    # 2️⃣ Parse the remaining JSON string into a list of URLs
    try:
        url_list: List[str] = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in URLs input: {e}")

    # 3️⃣ Extract content using TavilyClient
    response = client.extract(urls=url_list)

    extracted_content = []
    for item in response.get('results', []):
        url = item.get('url', '')
        content = item.get('raw_content', '')
        title = item.get('title', '')
        if not content:
            print(f"No content extracted for URL: {url}")
        extracted_content.append({
            "url": url,
            "title": title,
            "raw_content": content
        })

    # 4️⃣ Return as StepOutput with JSON string (for next .content usage)
    return extracted_content


# def extract_content_from_urls(urls) -> Dict[str, str]:
#     """
#     Extracts content from a list of URLs using TavilyClient.

#     Args:
#         urls (List[str]): A list of URLs to extract content from.

#     Returns:
#         Dict[str, str]: A dictionary mapping each URL to its extracted content.
#     """
#     # 1️⃣ Remove any Markdown code fences or the word 'json'
#     cleaned = re.sub(r'```|json', '', urls.content).strip()

#     # 2️⃣ Parse the remaining JSON string
#     urls = json.loads(cleaned)
#     print(urls)
#     response = client.extract(urls=urls)
#     extracted_content = []
#     for item in response['results']:
#         url = item['url']
#         content = item.get('raw_content', '')
#         title = item.get('title', '')
#         if not content:
#             print(f"No content extracted for URL: {url}")
#         extracted_content.append({"url": url, "title": title, "raw_content": content})
#     return extracted_content


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
        # GoogleSearchTools(all=True),
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
        "- After collecting all articles, select at most 15 diverse articles covering various different aspects and sentiments from different sources.",
        "- Prepare a list 'urls' of urls from the selected articles",
        
        "RETURN THE LIST OF URLS IN JSON FORMAT",
        
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
        "{ 'url': <url>, 'title': <title>, 'content': <cleaned full text> }",
    ],
    markdown=False,
    debug_mode=True,
)

news_coverage_agent = Agent(
    name="News Coverage Investigator",
    role="Analyze the entire corpus of articles to find the core factual claim(s) and coverage gaps",
    # model=Gemini(id="gemini-2.0-flash-lite"),
    model=model1,
    instructions=[
        "You are a cross-article coverage analyst. Your job is to analyze ALL provided articles together.",
        "You will receive a list of articles; each article includes: url, title and content.",
        "EXECUTE THESE STEPS IN ORDER:",
        "STEP 1: READ ALL ARTICLES",
        "- Ensure you process every article in the input. At the start, assert and report the total number of articles received.",
        # "STEP 2: EXTRACT PER-ARTICLE FACTUAL CLAIMS (NO SUMMARIES)",
        # "- For each article, extract discrete factual claims (3-10 per article if present), list direct quotes, and list topical tags.",
        # "- For each claim, include supporting evidence phrases (short snippets) and a confidence score 0.0–1.0.",
        "STEP 2: SYNTHESIZE CORPUS-LEVEL CORE IDEA",
        "- Using all per-article claim lists, infer the single core factual claim that the majority of articles support (short and precise paragraph).",
        "- Compute `agreement_pct` = percentage of articles that support that core claim (by checking claim presence).",
        "- List `supporting_articles` (urls) that support the claim and `contradicting_articles` if any.",
        "STEP 3: IDENTIFY COVERAGE GAPS",
        "- Identify deviating themes (coverage gaps) — distinct factual narratives or emphases present in subsets of articles.",
        "- For each gap, provide: gap_id, description about deviation from core news, pct_of_articles, and article urls.",
        "FINAL OUTPUT:",
        "- Return a complete report in JSON that includes:",
        "  * input_count and assertion that all articles were processed",
        "  * core_idea + agreement_pct + supporting_article list",
        "  * coverage_gaps with descriptions and percentages",
        
        "**DO NOT RETURN ANY OTHER TEXT. JUST RETURN VALID JSON REPORT NOTHING ELSE IS ALLOWED.**"
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
        "- For each component, produce a numeric sub-score and a short explanation (1–2 lines). Normalize to a 0–100 final credibility_score.",
        "STEP 3: AGGREGATE & RANK",
        "- After scoring all articles, produce a `top_10` list sorted by credibility_score (include id, title, source, date, score).",
        "FINAL OUTPUT:",
        "- Return a JSON report per article including: id, title, source, credibility_score, sub-score breakdown and short rationale.",

        "**DO NOT RETURN ANY OTHER TEXT. JUST RETURN VALID JSON REPORT NOTHING ELSE IS ALLOWED.**"
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
        "- If classification is uncertain for some articles, mark them with `needs_review` and continue."

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
    "- For each mention include: canonical_name, mention_count_in_article, quoted_mentions_count, and example quote (if quoted).",
    
    "STEP 2: MAP TO CANONICAL BUCKETS",
    "- Map extracted entities into canonical stakeholder buckets: Politicians, Farmers, Students, Businesses, NGOs, Journalists, Law Enforcement, Others.",
    "- If mapping is ambiguous, include internal resolution but do NOT output mapping_confidence or uncertainty scores.",
    
    "STEP 3: AGGREGATE SHARE-OF-VOICE",
    "- Across the corpus compute total_mentions and quoted_mentions per bucket and convert to percentages summing to 100%.",
    "- For each bucket include: total_mentions, quoted_mentions, percentage, and examples (up to 5 representative stakeholder mentions).",
    "- Each example must include canonical_name, mention_count_in_article, quoted_mentions_count, example_quote, and article_id (URL).",
    
    "STEP 4: COMPUTE TOP ENTITIES",
    "- Identify the top 10 most-mentioned canonical entities across all articles.",
    "- For each entity include: canonical_name, total_mentions, and example_article_ids (list of up to 7 URLs).",
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
    '          "total_mentions": integer,',
    '          "quoted_mentions": integer,',
    '          "percentage": float,',
    '          "examples": [',
    '              {',
    '                  "canonical_name": string,',
    '                  "mention_count_in_article": integer,',
    '                  "quoted_mentions_count": integer,',
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
    '          "total_mentions": integer,',
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
    '          "total_mentions": 33,',
    '          "quoted_mentions": 10,',
    '          "percentage": 42.31,',
    '          "examples": [',
    '              {',
    '                  "canonical_name": "Donald Trump",',
    '                  "mention_count_in_article": 14,',
    '                  "quoted_mentions_count": 5,',
    '                  "example_quote": "US President Donald Trump on Thursday expressed that he is likely to visit India next year while praising Prime Minister Narendra Modi and calling him a ‘great man’.",',
    '                  "article_id": "https://www.outlookbusiness.com/economy-and-policy/us-president-trump-likely-to-visit-india-next-year-calls-pm-modi-his-friend-and-a-great-man"',
    '              }',
    '          ]',
    '      },',
    '      "Farmers": { "total_mentions": 0, "quoted_mentions": 0, "percentage": 0.00, "examples": [] }',
    '  },',
    '  "top_10_entities": [',
    '      {',
    '          "canonical_name": "Donald Trump",',
    '          "total_mentions": 14,',
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

def news_pipeline(query: str):
    """
    Executes the full news analysis pipeline on the given query.
    
    Parameters:
        query (str): The search topic or query string.
    
    Returns:
        list[dict]: List of JSON objects containing results from all agents,
                    each labeled with the corresponding agent name.
    """
    # Step 1: Run search agent
    search_result = news_search_agent.run(query).content
    
    # Step 2: Extract content from URLs
    extracted_articles = extract_content_from_urls(search_result)
    
    # Step 3: Filter relevant articles
    filtered_articles = news_filter_agent.run(extracted_articles).content
    
    # Step 4: Run parallel analysis agents
    news_coverage = news_coverage_agent.run(filtered_articles).content
    credibility_score = credibility_agent.run(filtered_articles).content
    sentiment_res = sentiment_agent.run(filtered_articles).content
    voice_share = share_of_voice_agent.run(filtered_articles).content

    # Step 5: Convert string outputs to JSON safely
    def safe_json_loads(data, agent_name):
        try:
            if data.strip().startswith("```json") or data.strip().startswith("json```"):
            # Remove any code fences or 'json' markers
                cleaned = re.sub(r'```|json', '', data, flags=re.IGNORECASE).strip()
                return json.loads(cleaned)
            return json.loads(data)
        except Exception as e:
            return {"agent": agent_name, "error": str(e), "raw_output": data}

    news_coverage_json = {"agent": "news_coverage_agent", "data": safe_json_loads(news_coverage, "news_coverage_agent")}
    credibility_json = {"agent": "credibility_agent", "data": safe_json_loads(credibility_score, "credibility_agent")}
    sentiment_json = {"agent": "sentiment_agent", "data": safe_json_loads(sentiment_res, "sentiment_agent")}
    voice_share_json = {"agent": "share_of_voice_agent", "data": safe_json_loads(voice_share, "share_of_voice_agent")}
    
    # Step 6: Merge all results into a list of JSONs
    results = [
        news_coverage_json,
        credibility_json,
        sentiment_json,
        voice_share_json
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"news_pipeline_results_{timestamp}.json"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"[💾] Results saved to {save_path}")
    except Exception as e:
        print(f"[❌] Failed to save results: {e}")
    
    return results


# Create AgentOS with your workflow
result=news_pipeline("trump tarrifs over india")
print(result)