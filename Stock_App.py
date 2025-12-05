import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
import requests
import time
import datetime
from datetime import timedelta
import feedparser
import ast
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Institutional Committee", layout="wide", page_icon="⚖️")

# --- SESSION STATE INITIALIZATION ---
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "identifier_results" not in st.session_state:
    st.session_state.identifier_results = None
if "health_result" not in st.session_state:
    st.session_state.health_result = None
if "movers_result" not in st.session_state:
    st.session_state.movers_result = None
if "earnings_calendar" not in st.session_state:
    st.session_state.earnings_calendar = None
if "earnings_analysis_queue" not in st.session_state:
    st.session_state.earnings_analysis_queue = [] 
if "horizon" not in st.session_state:
    st.session_state.horizon = "Medium" 
if "identifier_limit" not in st.session_state:
    st.session_state.identifier_limit = None
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "manual" 

# --- UNIVERSE GENERATOR ---
@st.cache_data(ttl=3600*24)
def get_sp500_tickers():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return df['Symbol'].tolist()
    except Exception as e:
        st.error(f"CRITICAL ERROR: Could not fetch S&P 500 list from GitHub.\nReason: {e}")
        return []

def get_relevant_tickers(client, topic):
    """Generates a list of tickers based on interest/industry."""
    if not client:
        st.error("OpenAI Client is required.")
        return []

    try:
        prompt = f"""
        Act as a Senior Market Strategist.
        TASK: Identify a list of stock tickers (or ETFs) that offer the best investment exposure to this theme: "{topic}".
        GUIDELINES: Relevance is Key. Select assets genuinely driven by this theme.
        OUTPUT: Return ONLY a valid Python list of ticker strings. Max 20 tickers.
        Example: ['TICK1', 'TICK2']
        """
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        content = response.output_text.strip().replace("```python", "").replace("```", "").strip()
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match: content = match.group(0)
        tickers = ast.literal_eval(content)
        return tickers if isinstance(tickers, list) else []
    except Exception as e:
        st.error(f"Identifier Error: {e}")
        return []

# --- NEWS ENGINE ---
def get_google_news(query, limit=5):
    """Standard Financial News Fetcher"""
    try:
        encoded_query = requests.utils.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        return parse_rss(rss_url, limit)
    except Exception as e:
        print(f"News Fetch Error: {e}")
        return []

def parse_rss(url, limit):
    """Helper to parse RSS feeds with proper headers"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=5)
        feed = feedparser.parse(response.content)
        
        valid_articles = []
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
        
        for entry in feed.entries:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed))
                if pub_date > cutoff_date:
                    clean_date = pub_date.strftime('%Y-%m-%d')
                    title = entry.title.split(" - ")[0]
                    source = "Unknown"
                    if hasattr(entry, 'source'):
                        if isinstance(entry.source, dict):
                            source = entry.source.get('title', 'Unknown')
                        elif hasattr(entry.source, 'title'):
                            source = entry.source.title
                    valid_articles.append(f"- {title} ({source}, {clean_date})")
        return valid_articles[:limit]
    except Exception as e:
        print(f"RSS Parsing Error: {e}")
        return []

def get_news_sentiment_score(client, ticker, news_items):
    if not news_items: return 5.0
    headlines = " | ".join(news_items)
    prompt = f"Rate sentiment of {ticker} based on headlines: '{headlines}'. Return score 0-10 only (5 is neutral). Return ONLY the number."
    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        match = re.search(r"\d+(\.\d+)?", response.output_text)
        if match: return float(match.group(0))
        return 5.0
    except: return 5.0

# --- TECHNICAL ANALYSIS ENGINE ---
def calculate_technical_score(df, horizon):
    try:
        df = df.dropna()
        if len(df) < 50: return 0, None
        
        close = df['Close']
        curr_price = close.iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1]
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.00001)
        rsi = 100 - (100 / (1 + (gain/loss))).iloc[-1]

        score = 0
        if curr_price > sma_200: score += 20
        if curr_price > sma_50: score += 20

        if horizon == "Short": 
            if rsi < 30: score += 60
            elif rsi < 40: score += 40
            elif rsi < 50: score += 20
            elif rsi > 70: score -= 20
            if curr_price < sma_200 and rsi < 25: score += 50
        elif horizon == "Medium": 
            if curr_price > sma_50:
                if 50 < rsi < 70: score += 60
                elif rsi > 70: score += 40
                elif rsi < 40: score -= 10
        elif horizon == "Long": 
            if sma_50 > sma_200: 
                score += 40
                dist = abs((curr_price - sma_50)/sma_50 * 100)
                if dist < 5: score += 20
        
        metrics = {
            "price": curr_price,
            "rsi": rsi,
            "sma50": sma_50,
            "sma200": sma_200
        }
        return score, metrics
    except:
        return 0, None

# --- MARKET HEALTH ENGINE ---
def run_market_health_check(client, progress_bar):
    progress_bar.write("🚑 Connecting to Market Vital Monitors (SPY, VIX, TNX)...")
    try:
        tickers = ["SPY", "^VIX", "^TNX"] 
        data = yf.download(tickers, period="6mo", group_by='ticker', progress=False)
        
        def get_last(tick):
            try:
                series = data[tick]['Close']
                val = series.dropna().iloc[-1]
                return float(val)
            except Exception as e:
                try: return float(data['Close'].iloc[-1]) 
                except: return 0.0
        
        spy_price = get_last("SPY")
        vix_price = get_last("^VIX")
        tnx_price = get_last("^TNX")
        
        spy_hist = data["SPY"]['Close'].dropna()
        spy_sma50 = spy_hist.rolling(50).mean().iloc[-1]
        spy_sma200 = spy_hist.rolling(200).mean().iloc[-1]
        
    except Exception as e:
        return {"error": f"Data Fetch Error: {e}"}

    progress_bar.write("🌍 Scanning Global Wires for Fed, Geopolitics, and Economy...")
    macro_news = get_google_news("US Stock Market Economy Federal Reserve Geopolitics", limit=10)
    news_str = "\n".join(macro_news)

    progress_bar.write("🧠 Chief Strategist Synthesizing Data...")
    prompt = f"""
    You are a Senior Global Macro Strategist. It is Monday Morning. 
    Perform a Market Health Check based on this data:
    
    [TECHNICAL VITAL SIGNS]
    - S&P 500 (SPY): ${spy_price:.2f} (vs 50SMA: ${spy_sma50:.2f} | 200SMA: ${spy_sma200:.2f})
    - VIX (Fear Index): {vix_price:.2f} (High > 20 is Fear, Low < 13 is Complacency)
    - 10-Year Treasury Yield (TNX): {tnx_price:.2f}% (High yields hurt tech/growth)
    
    [MACRO NEWS WIRE]
    {news_str}
    
    TASK:
    1. Assign a "Market Health Score" from 0 (Market Crash/Extreme Bear) to 100 (Euphoria/Extreme Bull).
    2. Provide a "Weekly Outlook": Bullish, Bearish, or Neutral/Choppy.
    3. "Educated Estimation": Do not just say "we don't know". Based on the VIX and Fed news, what is the PROBABLE direction?
    4. List 3 Key Catalysts (e.g. Fed Rate Decision, War tensions, Tech Earnings) driving this week.
    
    FORMAT:
    Return valid JSON format ONLY:
    {{
        "score": integer,
        "outlook": "string",
        "rationale": "string",
        "catalysts": ["string1", "string2", "string3"],
        "summary": "string"
    }}
    """
    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        content = response.output_text.strip().replace("```json", "").replace("```", "").strip()
        result = ast.literal_eval(content)
        result['spy'] = spy_price
        result['vix'] = vix_price
        result['tnx'] = tnx_price
        return result
    except Exception as e:
        return {"error": f"AI Analysis Failed: {e}"}

# --- EXPLAIN THE ACTION ENGINE ---
def get_top_movers_sp500(progress_bar):
    progress_bar.write("📡 Scanning S&P 500 for biggest movers (This takes ~15s)...")
    tickers = get_sp500_tickers()
    if not tickers: return [], []
    
    try:
        data = yf.download(tickers, period="2d", group_by='ticker', progress=False, threads=True)
    except Exception as e:
        return [], []
    
    performance = []
    for ticker in tickers:
        try:
            if len(tickers) > 1: df = data[ticker]
            else: df = data
            closes = df['Close'].dropna()
            if len(closes) >= 2:
                prev_close = float(closes.iloc[-2])
                curr_close = float(closes.iloc[-1])
                pct_change = ((curr_close - prev_close) / prev_close) * 100
                performance.append({"ticker": ticker, "change": pct_change, "price": curr_close})
        except: continue
        
    performance.sort(key=lambda x: x['change'], reverse=True)
    return performance[:3], performance[-3:]

def analyze_price_action(client, stock_data, is_gain):
    ticker = stock_data['ticker']
    change = stock_data['change']
    price = stock_data['price']
    direction = "GAIN" if is_gain else "LOSS"
    
    news = get_google_news(f"{ticker} stock news", limit=4)
    news_str = "\n".join(news) if news else "No specific news headlines found."
    
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector', 'Unknown')
        pe = info.get('trailingPE', 'N/A')
    except: sector, pe = "Unknown", "N/A"
        
    prompt = f"""
    You are a Financial Action Analyst. 
    Stock: {ticker} ({sector})
    Move: {direction} of {change:.2f}% today. Current Price: ${price:.2f}.
    P/E Ratio: {pe}
    [NEWS HEADLINES]: {news_str}
    
    TASK:
    1. EXPLAIN THE ACTION: Why did it move?
    2. TOMORROW'S FORECAST: Based on momentum/panic, what happens tomorrow? 
    3. PRICE TARGETS: Give specific Bull/Bear targets for tomorrow.
    
    FORMAT JSON: {{ "explanation": "string", "forecast": "string", "bull_target": number, "bear_target": number }}
    """
    try:
        response = client.responses.create(model="gpt-5-nano", input=prompt)
        content = response.output_text.strip().replace("```json", "").replace("```", "").strip()
        result = ast.literal_eval(content)
        result.update(stock_data)
        return result
    except:
        return {"explanation": "Analysis failed.", "forecast": "Unknown", "bull_target": 0, "bear_target": 0, **stock_data}

# --- WEEKLY EARNINGS ENGINE (NEW - SCRAPER VERSION) ---
def get_next_trading_days(start_date, n_days=5):
    """Returns a list of the next N trading days (skipping weekends)."""
    trading_days = []
    current_date = start_date
    while len(trading_days) < n_days:
        if current_date.weekday() < 5: # Mon-Fri are 0-4
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    return trading_days

def get_weekly_earnings_calendar(progress_bar):
    """Scrapes Yahoo Finance Calendar for all tickers, finds next 5 trading days."""
    today = datetime.datetime.now().date()
    target_days = get_next_trading_days(today, 5)
    
    progress_bar.write(f"📅 Scanning Yahoo Finance Calendar for next 5 trading days: {target_days[0]} to {target_days[-1]}...")
    
    earnings_map = {}
    
    # Session for robust requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    for d in target_days:
        date_str = d.strftime('%Y-%m-%d')
        display_day = d.strftime("%A")
        earnings_map[d] = {"morning": [], "evening": []}
        
        # Scrape Yahoo Finance for this specific day
        url = f"https://finance.yahoo.com/calendar/earnings?day={date_str}"
        
        try:
            # We use pandas read_html which is very powerful for tables
            # It needs lxml or html5lib installed in environment usually, or bs4
            response = session.get(url)
            dfs = pd.read_html(response.text)
            
            if not dfs: continue
            df = dfs[0] # The main earnings table
            
            # Columns usually: Symbol, Company, Earnings Call Time, EPS Estimate, Reported EPS, Surprise(%)
            # We need Symbol and Time
            
            if df.empty: continue
            
            # Batch fetch market caps for sorting
            # Get list of symbols from this page
            symbols = df['Symbol'].tolist()
            
            # Clean symbols (remove dots for yfinance)
            clean_symbols = [s.replace('.', '-') for s in symbols]
            
            # Fetch info in batch to get Market Caps
            # We chunk it to avoid URL length issues
            chunk_size = 100
            symbol_cap_map = {}
            
            for i in range(0, len(clean_symbols), chunk_size):
                chunk = clean_symbols[i:i+chunk_size]
                try:
                    tickers = yf.Tickers(" ".join(chunk))
                    for sym in chunk:
                        try:
                            # Tickers.tickers is a dict
                            info = tickers.tickers[sym].info
                            cap = info.get('marketCap', 0)
                            symbol_cap_map[sym] = cap
                        except:
                            symbol_cap_map[sym] = 0
                except: pass
            
            # Process row by row
            for index, row in df.iterrows():
                try:
                    ticker = row['Symbol'].replace('.', '-')
                    time_str = str(row.get('Earnings Call Time', 'Unknown'))
                    name = row.get('Company', ticker)
                    
                    mkt_cap = symbol_cap_map.get(ticker, 0)
                    
                    item = {"ticker": ticker, "name": name, "cap": mkt_cap, "date": d}
                    
                    # Sort into Morning/Evening based on string
                    # Yahoo usually says "After Market Close", "Before Market Open" or specific times
                    lower_time = time_str.lower()
                    if "after" in lower_time or "pm" in lower_time or "close" in lower_time:
                         earnings_map[d]["evening"].append(item)
                    else:
                         # Default to morning for "Before Market Open" or unknown/unsupplied times
                         earnings_map[d]["morning"].append(item)
                         
                except: continue
                
        except Exception as e:
            # If no earnings for that day or error, just skip
            print(f"Error scraping {date_str}: {e}")
            continue

    # Sort and Finalize
    sorted_calendar = []
    for d in target_days:
        earnings_map[d]["morning"].sort(key=lambda x: x['cap'], reverse=True)
        earnings_map[d]["evening"].sort(key=lambda x: x['cap'], reverse=True)
        
        # Only add days that have data
        if earnings_map[d]["morning"] or earnings_map[d]["evening"]:
             sorted_calendar.append({"date": d, "day": d.strftime("%A"), "data": earnings_map[d]})
            
    return sorted_calendar

def predict_earnings_outcome(client, ticker):
    """Deep dive analysis into earnings prediction."""
    # 1. Get Data
    news = get_google_news(f"{ticker} earnings news forecast", limit=6)
    news_str = "\n".join(news)
    
    try:
        t = yf.Ticker(ticker)
        info = t.info
        est_eps = info.get('trailingEps', 'N/A') 
        rev = info.get('totalRevenue', 'N/A')
        sector = info.get('sector', 'Unknown')
    except:
        est_eps, rev, sector = "N/A", "N/A", "Unknown"

    prompt = f"""
    You are an Earnings Prediciton Expert.
    Stock: {ticker} ({sector}) is reporting earnings soon.
    
    [NEWS & SENTIMENT]:
    {news_str}
    
    [FINANCIAL CONTEXT]:
    EPS (Trailing): {est_eps}
    Revenue: {rev}
    
    TASK:
    1. INVESTOR EXPECTATIONS: What does the market want to see? (High growth, cost cutting, AI guidance?)
    2. YOUR PREDICTION: Based on the news clues, will they BEAT or MISS? (Positive/Negative Surprise).
    3. PRICE REACTION: Do you expect a "Gap Up", "Gap Down", or "Flat" move?
    4. KEY METRICS: Predict likely revenue trajectory (Up/Down).
    
    FORMAT JSON:
    {{
        "expectations": "string",
        "prediction": "BEAT" or "MISS",
        "rationale": "string",
        "price_move": "Gap Up/Down/Flat",
        "metrics": "string"
    }}
    """
    try:
        response = client.responses.create(model="gpt-5-nano", input=prompt)
        content = response.output_text.strip().replace("```json", "").replace("```", "").strip()
        result = ast.literal_eval(content)
        result['ticker'] = ticker
        return result
    except Exception as e:
        return {"ticker": ticker, "prediction": "ERROR", "rationale": str(e), "expectations": "N/A", "price_move": "N/A", "metrics": "N/A"}

# --- WORKFLOWS ---
def run_quantitative_dragnet(tickers, horizon, progress_bar):
    if not tickers: return []
    progress_bar.write(f"📥 Downloading data for {len(tickers)} companies...")
    try:
        data = yf.download(tickers, period="1y", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        st.error(f"YFinance Download Error: {e}")
        return []

    candidates = []
    progress_bar.write("🧮 Grading stocks using Weighted Scorecard...")

    for ticker in tickers:
        try:
            if len(tickers) == 1: df = data
            else: df = data[ticker]
            score, metrics = calculate_technical_score(df, horizon)
            if score > 40 and metrics:
                candidate = {"ticker": ticker, "math_score": score}
                candidate.update(metrics)
                candidates.append(candidate)
        except: continue
    candidates.sort(key=lambda x: x['math_score'], reverse=True)
    return candidates

def run_junior_analyst_batch(client, candidates, progress_bar):
    progress_bar.write(f"🕵️ Junior Analyst scanning headlines for Top {len(candidates)}...")
    scored_candidates = []
    for cand in candidates:
        ticker = cand['ticker']
        news_items = get_google_news(f"{ticker} stock news", limit=3)
        sentiment_score = get_news_sentiment_score(client, ticker, news_items)
        
        if sentiment_score < 3.0: sentiment_impact = -50 
        else: sentiment_impact = (sentiment_score - 5) * 1 
        
        cand['final_score'] = cand['math_score'] + sentiment_impact
        cand['headlines'] = " | ".join(news_items)
        scored_candidates.append(cand)
        
    scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return scored_candidates[0]

def rank_identified_stocks(client, tickers, horizon, progress_bar):
    progress_bar.write(f"📥 Fetching data for {len(tickers)} identified stocks...")
    try:
        data = yf.download(tickers, period="1y", group_by='ticker', threads=True, progress=False)
    except: return []

    ranked_candidates = []
    progress_bar.write("📊 Calculating Ranking Scores (Technical + News Sentiment)...")
    
    for i, ticker in enumerate(tickers):
        try:
            if len(tickers) == 1: df = data
            else: df = data[ticker]
            math_score, metrics = calculate_technical_score(df, horizon)
            if not metrics: continue 
            
            news_items = get_google_news(f"{ticker} stock news", limit=3)
            sent_score = get_news_sentiment_score(client, ticker, news_items)
            
            if sent_score < 3.0: sentiment_impact = -50 
            else: sentiment_impact = (sent_score - 5) * 1 
            
            total_score = math_score + sentiment_impact
            cand = {
                "ticker": ticker,
                "total_score": total_score,
                "math_score": math_score,
                "sent_score": sent_score,
                "headlines": " | ".join(news_items)
            }
            cand.update(metrics)
            ranked_candidates.append(cand)
        except: continue
    ranked_candidates.sort(key=lambda x: x['total_score'], reverse=True)
    return ranked_candidates

def run_committee_meeting(openai_client, gemini_key, winner_data, horizon, allow_sell=False, topic_interest=None):
    ticker = winner_data['ticker']
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    
    try:
        info = stock.info
        sector = info.get('sector', 'Unknown')
        pe_ratio = info.get('trailingPE', 'N/A')
        fwd_pe = info.get('forwardPE', 'N/A')
        beta = info.get('beta', 'N/A')
    except: sector, pe_ratio, fwd_pe, beta = "Unknown", "N/A", "N/A", "N/A"

    news_list = get_google_news(f"{ticker} stock news", limit=5)
    full_news_text = "\n".join(news_list) if news_list else ""
    peer_str, peer_list, avg_peer_pe = get_peer_context(openai_client, ticker)
    market_regime = get_market_regime()

    interest_context = ""
    if topic_interest:
        interest_context = f"\n[USER INTEREST ALIGNMENT]: The user is specifically interested in '{topic_interest}'. Evaluate how this stock fits this specific theme."

    context_data = f"""
    SUBJECT: {ticker}
    Sector: {sector}
    Price: ${winner_data['price']:.2f}
    RSI: {winner_data['rsi']:.1f}
    Trailing P/E: {pe_ratio}
    Forward P/E: {fwd_pe}
    Beta: {beta}
    Price vs 50 SMA: {"Above" if winner_data['price'] > winner_data['sma50'] else "Below"}
    [MARKET REGIME]: {market_regime}
    [PEER BENCHMARK]: {peer_str}
    {interest_context}
    """

    if full_news_text:
        news_prompt = f"""
        You are a Financial News Anchor.
        Extract purely factual material events (Earnings, Products, Legal).
        Headlines: {full_news_text}
        Task: List top 3 events with dates. NO Opinions.
        """
        news_briefing = openai_client.responses.create(
            model="gpt-5-nano",
            input=news_prompt
        ).output_text
    else:
        news_briefing = "No significant news headlines found in the last 30 days."

    try:
        bear_prompt = f"""
        Role: Risk Analyst (Bear).
        Analyze {ticker} based on:
        {context_data}
        [NEWS]: {news_briefing}
        
        CRITICAL: Do NOT complain about high PE if it aligns with Peers.
        Task: Identify 3 risks (Valuation, Macro, Fundamental, and specifically Risks related to User Interest if applicable).
        Tone: Skeptical.
        """
        bear_argument = openai_client.responses.create(
            model="gpt-5-nano",
            input=bear_prompt
        ).output_text
    except Exception as e:
        bear_argument = f"Risk Analysis Unavailable: {e}"

    bull_prompt = f"""
    Role: Growth Strategist (Bull).
    Analyze {ticker} based on:
    {context_data}
    [NEWS]: {news_briefing}
    
    CRITICAL: If PE is high but lower than peers, call it "Relative Value".
    Task: Identify 3 catalysts (Including specific relevance/upside to the User Interest '{topic_interest}' if applicable). 
    Tone: Opportunistic.
    """
    bull_argument = openai_client.responses.create(
        model="gpt-5-nano",
        input=bull_prompt
    ).output_text

    verdict_instruction = "1. Verdict (BUY / SELL / WAIT)." if allow_sell else "1. Verdict (BUY / WAIT)."
    cio_prompt = f"""
    Role: Chief Investment Officer.
    
    [FACTS]: {news_briefing}
    [BEAR]: {bear_argument}
    [BULL]: {bull_argument}
    
    [FULL CONTEXT]:
    {context_data}
    
    Task: Final Execution Order.
    {verdict_instruction}
    2. Limit Price.
    3. Stop Loss.
    4. Rationale (Synthesize Peer Valuation, Market Trend, News Facts, AND User Interest Alignment).
    
    Tone: Directive.
    """
    try:
        response = openai_client.responses.create(
            model="gpt-5-nano",
            input=cio_prompt
        )
        final_verdict = response.output_text
    except Exception as e:
        final_verdict = f"CIO Error: {str(e)}"
    
    return {
        "raw_news": full_news_text, "news_briefing": news_briefing, "bear": bear_argument,
        "bull": bull_argument, "cio": final_verdict, "context_str": context_data, "topic": topic_interest
    }, hist

def get_market_regime():
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="6mo")
        curr = hist['Close'].iloc[-1]
        sma50 = hist['Close'].rolling(50).mean().iloc[-1]
        trend = "Bullish" if curr > sma50 else "Bearish"
        return f"S&P 500 Trend: {trend} (Price ${curr:.2f} vs 50SMA ${sma50:.2f})"
    except:
        return "Market Regime: Unknown"

def get_peer_context(client, ticker):
    try:
        prompt = f"""
        Return a Python list of strings containing the most relevant US-listed competitor tickers for {ticker}.
        Format: ['A', 'B', 'C']. No markdown.
        """
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        content = response.output_text.strip().replace("```python", "").replace("```", "").strip()
        peers = ast.literal_eval(content)
        if not isinstance(peers, list): peers = []
        
        peers = [p for p in peers if p.upper() != ticker.upper()]
        stats = []
        valid_peers_log = []
        
        for p in peers:
            try:
                stock_obj = yf.Ticker(p)
                info = stock_obj.info
                mkt_cap = info.get('marketCap', 0)
                if mkt_cap < 1_000_000_000: continue 
                
                pe = info.get('trailingPE', None)
                fwd_pe = info.get('forwardPE', None)
                stats.append({'ticker': p, 'pe': pe, 'fwd_pe': fwd_pe, 'cap': mkt_cap})
                valid_peers_log.append(p)
            except: continue
            
        valid_pe = [s['pe'] for s in stats if s['pe'] is not None]
        avg_pe = sum(valid_pe)/len(valid_pe) if valid_pe else "N/A"
        
        context_str = f"Peer Group: {', '.join(valid_peers_log)}.\n"
        if avg_pe != "N/A": context_str += f"Sector Avg PE: {avg_pe:.2f}.\n"
        
        return context_str, valid_peers_log, avg_pe
    except Exception as e:
        return f"Peer Error: {e}", [], "N/A"

# --- UI LOGIC ---
with st.sidebar:
    st.header("🔑 Credentials")
    openai_key = st.text_input("OpenAI API Key", type="password")
    
    st.markdown("---")
    st.header("🏥 Market Health")
    if st.button("Check Market Vitals"):
        st.session_state.active_mode = "health"
        st.session_state.analysis_result = None
        st.session_state.identifier_results = None
        st.session_state.movers_result = None
        st.session_state.earnings_calendar = None

    if st.button("Explain the Action (Movers)"):
        st.session_state.active_mode = "movers"
        st.session_state.analysis_result = None
        st.session_state.identifier_results = None
        st.session_state.health_result = None
        st.session_state.earnings_calendar = None
    
    if st.button("📅 Weekly Earnings"):
        st.session_state.active_mode = "earnings"
        st.session_state.analysis_result = None
        st.session_state.identifier_results = None
        st.session_state.health_result = None
        st.session_state.movers_result = None

    st.markdown("---")
    st.header("🔎 Manual Analysis")
    manual_ticker = st.text_input("Enter Ticker (e.g. NVDA, TSLA)")
    if st.button("Evaluate Ticker"):
        st.session_state.active_mode = "manual"
        st.session_state['run_manual'] = True
        st.session_state['manual_ticker'] = manual_ticker.upper().strip()

    st.markdown("---")
    st.header("🎯 Identifier")
    interest_topic = st.text_area("Your Interest (e.g. 'Robotics', 'Uranium')")
    limit_input = st.text_input("Max Stocks to Analyze (Optional)", placeholder="Leave blank for default")

    if st.button("Identify & Analyze"):
        if interest_topic:
            st.session_state.active_mode = "identifier"
            st.session_state['run_identifier'] = True
            st.session_state['interest_topic'] = interest_topic
            if limit_input and limit_input.strip().isdigit():
                st.session_state['identifier_limit'] = int(limit_input)
            else:
                st.session_state['identifier_limit'] = None
        else:
            st.error("Please enter a topic.")

st.title("⚖️ Institutional Committee: S&P 500")

if not openai_key:
    st.warning("Please enter OpenAI Key.")
    st.stop()

client = OpenAI(api_key=openai_key)

# --- MODE: MARKET HEALTH ---
if st.session_state.active_mode == "health":
    st.header("🏥 Market Health & Macro Outlook")
    if st.button("Run Diagnostics"):
        status = st.status("Analyzing Market Vitals...", expanded=True)
        result = run_market_health_check(client, status)
        if "error" in result:
            status.update(label="Error in Analysis", state="error")
            st.error(result['error'])
        else:
            status.update(label="Diagnostics Complete", state="complete", expanded=False)
            st.session_state.health_result = result
            st.rerun()

    if st.session_state.health_result:
        res = st.session_state.health_result
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Market Health", res['score'])
        kpi2.metric("S&P 500", f"${res['spy']:.2f}")
        kpi3.metric("VIX", f"{res['vix']:.2f}")
        kpi4.metric("10Y Yield", f"{res['tnx']:.2f}%")
        
        st.subheader(f"Weekly Outlook: {res['outlook']}")
        st.info(f"**Rationale:** {res['rationale']}")
        st.write("### Catalysts")
        for cat in res['catalysts']: st.markdown(f"👉 {cat}")

# --- MODE: EARNINGS ---
elif st.session_state.active_mode == "earnings":
    st.header("📅 Weekly Earnings Calendar")
    st.caption("Comprehensive scan of ALL US stocks reporting in next 5 trading days.")
    
    if st.button("Scan Upcoming Earnings"):
        status = st.status("Fetching Yahoo Finance Calendar (Scraping)...", expanded=True)
        calendar = get_weekly_earnings_calendar(status)
        if not calendar:
            status.update(label="No major earnings found.", state="complete")
        else:
            status.update(label="Schedule Loaded", state="complete", expanded=False)
            st.session_state.earnings_calendar = calendar
            st.rerun()
            
    if st.session_state.earnings_calendar:
        cols = st.columns(len(st.session_state.earnings_calendar))
        
        for i, day_data in enumerate(st.session_state.earnings_calendar):
            with cols[i]:
                st.subheader(day_data['day'])
                st.caption(day_data['date'].strftime('%b %d'))
                
                if day_data['data']['morning']:
                    st.markdown("**☀️ Morning**")
                    for stock in day_data['data']['morning']:
                        b_col1, b_col2 = st.columns([2, 1])
                        b_col1.write(f"**{stock['ticker']}**")
                        if b_col2.button("Analyze", key=f"btn_m_{stock['ticker']}"):
                            with st.spinner(f"Predicting {stock['ticker']} earnings..."):
                                res = predict_earnings_outcome(client, stock['ticker'])
                                st.session_state.earnings_analysis_queue.insert(0, res)
                                st.rerun()

                if day_data['data']['evening']:
                    st.markdown("**🌙 Evening**")
                    for stock in day_data['data']['evening']:
                        b_col1, b_col2 = st.columns([2, 1])
                        b_col1.write(f"**{stock['ticker']}**")
                        if b_col2.button("Analyze", key=f"btn_e_{stock['ticker']}"):
                            with st.spinner(f"Predicting {stock['ticker']} earnings..."):
                                res = predict_earnings_outcome(client, stock['ticker'])
                                st.session_state.earnings_analysis_queue.insert(0, res)
                                st.rerun()
                                
        st.divider()
        if st.session_state.earnings_analysis_queue:
            st.subheader("🔮 Earnings Prediction Queue")
            for analysis in st.session_state.earnings_analysis_queue:
                with st.expander(f"Report: {analysis['ticker']} ({analysis['prediction']})", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prediction", analysis['prediction'], delta=analysis['price_move'])
                    c2.info(f"**Expectations:**\n{analysis['expectations']}")
                    c3.success(f"**Metrics:**\n{analysis['metrics']}")
                    st.markdown(f"**Rationale:** {analysis['rationale']}")

# --- MODE: MOVERS ---
elif st.session_state.active_mode == "movers":
    st.header("⚡ Explain the Action (S&P 500 Movers)")
    if st.button("Scan Market & Analyze"):
        status = st.status("Scanning Real-Time Data...", expanded=True)
        gainers, losers = get_top_movers_sp500(status)
        
        results_g = []
        results_l = []
        
        status.write("🧠 Analyzing price drivers...")
        for g in gainers:
            analysis = analyze_price_action(client, g, is_gain=True)
            results_g.append(analysis)
        for l in losers:
            analysis = analyze_price_action(client, l, is_gain=False)
            results_l.append(analysis)
            
        st.session_state.movers_result = {"gainers": results_g, "losers": results_l}
        status.update(label="Analysis Complete", state="complete", expanded=False)
        st.rerun()

    if st.session_state.movers_result:
        res = st.session_state.movers_result
        st.subheader("🚀 Top Gainers")
        cols_g = st.columns(3)
        for i, item in enumerate(res['gainers']):
            with cols_g[i]:
                st.metric(label=item['ticker'], value=f"${item['price']:.2f}", delta=f"{item['change']:.2f}%")
                st.info(item['explanation'])
                
        st.subheader("🩸 Top Losers")
        cols_l = st.columns(3)
        for i, item in enumerate(res['losers']):
            with cols_l[i]:
                st.metric(label=item['ticker'], value=f"${item['price']:.2f}", delta=f"{item['change']:.2f}%")
                st.error(item['explanation'])

# --- MODE: STANDARD ---
else:
    col1, col2, col3 = st.columns(3)
    def set_horizon(h):
        st.session_state['horizon'] = h
        st.session_state['run'] = True
        st.session_state.active_mode = "manual"

    if col1.button("🚀 Short Term"): set_horizon("Short")
    if col2.button("📈 Medium Term"): set_horizon("Medium")
    if col3.button("💰 Long Term"): set_horizon("Long")

    if st.session_state.get('run', False):
        horizon = st.session_state['horizon']
        status = st.status(f"The Committee is in session ({horizon} Term)...", expanded=True)
        tickers = get_sp500_tickers()
        if not tickers: st.stop()
        
        top_60 = run_quantitative_dragnet(tickers, horizon, status)
        if not top_60: st.stop()
            
        status.write(f"🤖 Junior Analyst scanning Top {len(top_60)}...")
        winner = run_junior_analyst_batch(client, top_60, status)
        status.write(f"🔔 Candidate Selected: **{winner['ticker']}**")
        
        status.write("🧠 Gathering Context...")
        debate_results, hist = run_committee_meeting(client, None, winner, horizon)
        
        status.update(label="Meeting Adjourned", state="complete", expanded=False)
        st.session_state['analysis_result'] = {"winner": winner, "debate": debate_results, "hist": hist}
        st.session_state['run'] = False
        st.rerun()

    if st.session_state.get('run_manual', False) and st.session_state.active_mode == "manual":
        ticker = st.session_state['manual_ticker']
        horizon = st.session_state['horizon']
        status = st.status(f"Analysing {ticker}...", expanded=True)
        try:
            status.write("📥 Fetching Market Data...")
            df = yf.download(ticker, period="1y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(ticker, axis=1, level=1)
                except: pass 
            
            score, metrics = calculate_technical_score(df, horizon)
            if not metrics: st.stop()
            
            candidate_data = {"ticker": ticker}
            candidate_data.update(metrics)
            
            status.write("🧠 Gathering Context...")
            debate_results, hist = run_committee_meeting(client, None, candidate_data, horizon, allow_sell=True)
            
            status.update(label="Analysis Complete", state="complete", expanded=False)
            st.session_state['analysis_result'] = {"winner": candidate_data, "debate": debate_results, "hist": hist}
            st.session_state['run_manual'] = False
            st.rerun()
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            st.session_state['run_manual'] = False

    if st.session_state.get('run_identifier', False) and st.session_state.active_mode == "identifier":
        topic = st.session_state['interest_topic']
        horizon = st.session_state['horizon']
        user_limit = st.session_state.get('identifier_limit', None)

        status = st.status(f"Surveying Market for: {topic}", expanded=True)
        try:
            status.write("🤖 Scanning Global Market...")
            identified_tickers = get_relevant_tickers(client, topic)
            if not identified_tickers: st.stop()
            
            status.write(f"🔎 Identified: {', '.join(identified_tickers)}")
            ranked_candidates = rank_identified_stocks(client, identified_tickers, horizon, status)
            if not ranked_candidates: st.stop()
            
            if user_limit is not None:
                ranked_candidates = ranked_candidates[:user_limit]
                status.write(f"✂️ Filtering top {user_limit} candidates.")
            
            results = []
            for i, candidate in enumerate(ranked_candidates):
                rank = i + 1
                tick = candidate['ticker']
                status.write(f"🧠 Deep Dive: {tick}...")
                debate, hist = run_committee_meeting(client, None, candidate, horizon, allow_sell=True, topic_interest=topic)
                results.append({"ticker": tick, "rank": rank, "debate": debate, "hist": hist})
                
            status.update(label="Survey Complete", state="complete", expanded=False)
            st.session_state['identifier_results'] = results
            st.session_state['run_identifier'] = False
            st.rerun()
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            st.session_state['run_identifier'] = False

    if st.session_state['analysis_result'] and st.session_state.active_mode == "manual":
        res = st.session_state['analysis_result']
        debate = res['debate']
        ticker = res['winner']['ticker']
        
        st.divider()
        st.header(f"Committee Transcript: {ticker}")
        st.info(debate['news_briefing'])

        cb, cbu = st.columns(2)
        with cb:
            st.subheader("🐻 The Bear")
            st.markdown(debate['bear']) 
        with cbu:
            st.subheader("🐂 The Bull")
            st.markdown(debate['bull']) 
            
        st.success(debate['cio'])
        
        hist = res['hist']
        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(50).mean(), line=dict(color='orange'), name='SMA 50'))
        fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.get('identifier_results') and st.session_state.active_mode == "identifier":
        st.divider()
        st.header(f"🎯 Market Survey Results")
        results = st.session_state['identifier_results']
        for res in results:
            ticker = res['ticker']
            rank = res['rank']
            debate = res['debate']
            with st.expander(f"#{rank} {ticker}", expanded=(rank==1)):
                st.success(debate['cio'])
