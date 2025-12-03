import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
import google.generativeai as genai_old # Renaming old SDK to avoid conflict
from google import genai # New SDK for Gemini 3
import requests
import time
import datetime
import feedparser
import ast
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Institutional Committee", layout="wide", page_icon="⚖️")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "identifier_results" not in st.session_state:
    st.session_state.identifier_results = None
if "horizon" not in st.session_state:
    st.session_state.horizon = "Medium" # Default

# --- UNIVERSE GENERATOR ---
@st.cache_data(ttl=3600*24)
def get_sp500_tickers():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].tolist()
    except Exception as e:
        st.error(f"CRITICAL ERROR: Could not fetch S&P 500 list from GitHub.\nReason: {e}")
        return []

def get_relevant_tickers(api_key, topic):
    """
    Generates a list of tickers based on a user's specific interest/industry.
    UPDATED: Uses Gemini 3 Pro Preview via the new google.genai SDK.
    """
    if not api_key:
        st.error("Gemini API Key is required for the Identifier.")
        return []

    try:
        # Initialize the new Client with the provided API key
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Act as a Senior Market Strategist.
        
        TASK:
        Identify a list of stock tickers (or ETFs) that offer the best investment exposure to this specific theme: "{topic}".
        
        GUIDELINES:
        1. **Relevance is Key**: Select assets that are genuinely driven by this theme.

        OUTPUT:
        - Return ONLY a valid Python list of ticker strings.
        - Max 20 tickers.
        - Example: ['TICK1', 'TICK2', 'TICK3', 'TICK4']
        - Do not use Markdown formatting like ```python. Just the raw list.
        """
        
        # Using the new Gemini 3 Pro Preview model syntax
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
        )
        
        content = response.text.strip()
        
        # Clean up potential markdown or extra text if the model is chatty
        content = content.replace("```python", "").replace("```", "").replace("\n", "").strip()
        
        # Robust parsing using Regex to find the list bracket structure
        match = re.search(r"\[.*\]", content)
        if match:
            content = match.group(0)
            
        tickers = ast.literal_eval(content)
        return tickers if isinstance(tickers, list) else []
    except Exception as e:
        st.error(f"Identifier Error (Gemini 3): {e}")
        return []

# --- NEWS ENGINES ---
def get_google_news(ticker, limit=5):
    """Standard Financial News Fetcher"""
    try:
        encoded_ticker = requests.utils.quote(f"{ticker} stock news")
        rss_url = f"[https://news.google.com/rss/search?q=](https://news.google.com/rss/search?q=){encoded_ticker}&hl=en-US&gl=US&ceid=US:en"
        return parse_rss(rss_url, limit)
    except Exception as e:
        return [f"News Error: {str(e)}"]

def get_sentiment_feed(ticker, limit=5):
    """Social Sentiment Fetcher."""
    try:
        query = f"{ticker} stock investor sentiment reddit discussion analysis"
        encoded_query = requests.utils.quote(query)
        rss_url = f"[https://news.google.com/rss/search?q=](https://news.google.com/rss/search?q=){encoded_query}&hl=en-US&gl=US&ceid=US:en"
        return parse_rss(rss_url, limit)
    except Exception as e:
        return [f"Sentiment Feed Error: {str(e)}"]

def parse_rss(url, limit):
    """Helper to parse RSS feeds"""
    feed = feedparser.parse(url)
    valid_articles = []
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
    
    for entry in feed.entries:
        if hasattr(entry, 'published_parsed'):
            pub_date = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed))
            if pub_date > cutoff_date:
                clean_date = pub_date.strftime('%Y-%m-%d')
                valid_articles.append(f"- {entry.title} ({entry.source.title}, {clean_date})")
    return valid_articles[:limit]

def get_sentiment_score(client, ticker, news_items):
    """Quick sentiment check for ranking purposes."""
    headlines = " | ".join(news_items) if news_items else "No recent news."
    prompt = f"Rate sentiment of {ticker} based on headlines: '{headlines}'. Return score 0-10 only (5 is neutral)."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )
        return float(response.choices[0].message.content.strip())
    except: 
        return 5.0

# --- TECHNICAL ANALYSIS ENGINE ---
def calculate_technical_score(df, horizon):
    """
    Calculates a technical score (0-100+) based on the horizon.
    Returns: (score, metrics_dict) or (0, None) if invalid.
    """
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

# --- CONTEXT LAYER ---
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
        - Format: ['A', 'B', 'C']. No markdown.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip().replace("```python", "").replace("```", "").strip()
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

# --- WORKFLOW: S&P 500 SCANNER ---
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
    """Scans news for the candidates and selects the winner."""
    progress_bar.write(f"🕵️ Junior Analyst scanning headlines for Top {len(candidates)}...")
    scored_candidates = []
    for cand in candidates:
        ticker = cand['ticker']
        news_items = get_google_news(ticker, limit=3)
        sentiment_score = get_sentiment_score(client, ticker, news_items)
        
        cand['final_score'] = cand['math_score'] + (sentiment_score * 5)
        cand['headlines'] = " | ".join(news_items)
        scored_candidates.append(cand)
        
    scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return scored_candidates[0] # Return the winner

# --- WORKFLOW: IDENTIFIER & RANKING ---
def rank_identified_stocks(client, tickers, horizon, progress_bar):
    """Fetches data, calculates scores, and ranks tickers found by the Identifier."""
    progress_bar.write(f"📥 Fetching data for {len(tickers)} identified stocks...")
    try:
        data = yf.download(tickers, period="1y", group_by='ticker', threads=True, progress=False)
    except: return []

    ranked_candidates = []
    progress_bar.write("📊 Calculating Ranking Scores (Technical + Sentiment)...")
    
    for i, ticker in enumerate(tickers):
        try:
            # 1. Get Dataframe
            if len(tickers) == 1: df = data
            else: df = data[ticker]
            
            # 2. Math Score
            math_score, metrics = calculate_technical_score(df, horizon)
            if not metrics: continue # Skip if no data or too short
            
            # 3. Sentiment Score
            news_items = get_google_news(ticker, limit=3)
            sent_score = get_sentiment_score(client, ticker, news_items)
            
            # 4. Total Score
            total_score = math_score + (sent_score * 5)
            
            cand = {
                "ticker": ticker,
                "total_score": total_score,
                "math_score": math_score,
                "sent_score": sent_score,
                "headlines": " | ".join(news_items)
            }
            cand.update(metrics)
            ranked_candidates.append(cand)
            
        except Exception as e:
            continue
            
    # Rank by Total Score
    ranked_candidates.sort(key=lambda x: x['total_score'], reverse=True)
    return ranked_candidates

# --- STAGE 3: THE COMMITTEE MEETING ---
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
    except: 
        sector, pe_ratio, fwd_pe, beta = "Unknown", "N/A", "N/A", "N/A"

    # DATA FETCHING
    news_list = get_google_news(ticker, limit=5)
    full_news_text = "\n".join(news_list) if news_list else "No recent news."
    
    sentiment_list = get_sentiment_feed(ticker, limit=5)
    full_sentiment_text = "\n".join(sentiment_list) if sentiment_list else "No recent sentiment data."

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

    # --- AGENT 1: NEWS ANCHOR ---
    news_prompt = f"""
    You are a Financial News Anchor.
    Extract purely factual material events (Earnings, Products, Legal).
    Headlines: {full_news_text}
    Task: List top 3 events with dates. NO Opinions.
    """
    news_briefing = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": news_prompt}],
        temperature=0.0
    ).choices[0].message.content

    # --- AGENT 2: SOCIAL SENTIMENT ANALYST ---
    sentiment_prompt = f"""
    You are a Social Sentiment Analyst.
    Your job is to analyze "soft data" (social media discussions, retail sentiment, viral topics) in a PROFESSIONAL, objective manner.
    
    Sources:
    {full_sentiment_text}
    
    Task:
    1. Identify key discussion topics.
    2. Quantify the vibe where possible.
    3. Strictly NO fluff.
    4. Focus purely on the stock: {ticker}.
    
    Tone: Institutional, concise, data-driven.
    """
    sentiment_report = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": sentiment_prompt}],
        temperature=0.3
    ).choices[0].message.content

    # --- AGENT 3: THE BEAR ---
    bear_argument = "Gemini API Error."
    if gemini_key:
        try:
            # Using the OLD SDK alias for this part to avoid breaking legacy code
            genai_old.configure(api_key=gemini_key)
            model = genai_old.GenerativeModel('gemini-2.5-flash')
            bear_prompt = f"""
            Role: Risk Analyst (Bear).
            Analyze {ticker} based on:
            {context_data}
            [NEWS]: {news_briefing}
            
            CRITICAL: Do NOT complain about high PE if it aligns with Peers.
            Task: Identify 3 risks (Valuation, Macro, Fundamental, and specifically Risks related to User Interest if applicable).
            Tone: Skeptical.
            """
            bear_argument = model.generate_content(bear_prompt).text
        except Exception as e:
            bear_argument = f"Risk Analysis Unavailable: {e}"

    # --- AGENT 4: THE BULL ---
    bull_prompt = f"""
    Role: Growth Strategist (Bull).
    Analyze {ticker} based on:
    {context_data}
    [NEWS]: {news_briefing}
    
    CRITICAL: If PE is high but lower than peers, call it "Relative Value".
    Task: Identify 3 catalysts (Including specific relevance/upside to the User Interest '{topic_interest}' if applicable). 
    Tone: Opportunistic.
    """
    bull_argument = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": bull_prompt}],
        temperature=0.1
    ).choices[0].message.content

    # --- AGENT 5: THE CIO ---
    verdict_instruction = "1. Verdict (BUY / SELL / WAIT)." if allow_sell else "1. Verdict (BUY / WAIT)."
    cio_prompt = f"""
    Role: Chief Investment Officer.
    
    [FACTS]: {news_briefing}
    [SENTIMENT/SOCIAL]: {sentiment_report}
    [BEAR]: {bear_argument}
    [BULL]: {bull_argument}
    
    [FULL CONTEXT]:
    {context_data}
    
    Task: Final Execution Order.
    {verdict_instruction}
    2. Limit Price.
    3. Stop Loss.
    4. Rationale (Synthesize Peer Valuation, Market Trend, Social Sentiment, AND User Interest Alignment).
    
    Tone: Directive.
    """
    try:
        response = openai_client.responses.create(model="gpt-5.1", input=cio_prompt)
        final_verdict = response.output_text
    except Exception as e:
        final_verdict = f"CIO Error: {str(e)}"
    
    return {
        "raw_news": full_news_text,
        "raw_sentiment": full_sentiment_text,
        "news_briefing": news_briefing,
        "sentiment_report": sentiment_report,
        "bear": bear_argument,
        "bull": bull_argument,
        "cio": final_verdict,
        "context_str": context_data,
        "topic": topic_interest
    }, hist

# --- UI LOGIC ---
with st.sidebar:
    st.header("🔑 Credentials")
    openai_key = st.text_input("OpenAI API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.header("🔎 Manual Analysis")
    manual_ticker = st.text_input("Enter Ticker (e.g. NVDA, TSLA)")
    if st.button("Evaluate Ticker"):
        st.session_state['run_manual'] = True
        st.session_state['manual_ticker'] = manual_ticker.upper().strip()
        st.session_state['run_identifier'] = False

    st.markdown("---")
    st.header("🎯 Identifier")
    interest_topic = st.text_area("Your Interest (e.g. 'Robotics with VLAMs', 'Uranium miners')")
    if st.button("Identify & Analyze"):
        if interest_topic:
            st.session_state['run_identifier'] = True
            st.session_state['interest_topic'] = interest_topic
            st.session_state['run_manual'] = False
        else:
            st.error("Please enter a topic.")

st.title("⚖️ Institutional Committee: S&P 500")

if not openai_key:
    st.warning("Please enter OpenAI Key.")
    st.stop()

client = OpenAI(api_key=openai_key)
col1, col2, col3 = st.columns(3)

def set_horizon(h):
    st.session_state['horizon'] = h
    st.session_state['run'] = True
    st.session_state['run_manual'] = False
    st.session_state['run_identifier'] = False

if col1.button("🚀 Short Term"): set_horizon("Short")
if col2.button("📈 Medium Term"): set_horizon("Medium")
if col3.button("💰 Long Term"): set_horizon("Long")

# --- AUTOMATIC S&P 500 WORKFLOW ---
if st.session_state.get('run', False):
    horizon = st.session_state['horizon']
    status = st.status(f"The Committee is in session ({horizon} Term)...", expanded=True)
    status.write("🌎 Accessing S&P 500 Database (GitHub)...")
    tickers = get_sp500_tickers()
    if not tickers:
        status.update(label="Critical Error: S&P 500 List not found.", state="error")
        st.stop()
    
    top_60 = run_quantitative_dragnet(tickers, horizon, status)
    if not top_60:
        st.error("No stocks met the criteria.")
        st.stop()
        
    status.write(f"🤖 Junior Analyst scanning Top {len(top_60)} for News Sentiment...")
    winner = run_junior_analyst_batch(client, top_60, status)
    status.write(f"🔔 Candidate Selected: **{winner['ticker']}**")
    
    status.write("🧠 Gathering Context & Sentiment...")
    debate_results, hist = run_committee_meeting(client, gemini_key, winner, horizon)
    
    status.update(label="Meeting Adjourned", state="complete", expanded=False)
    st.session_state['analysis_result'] = {"winner": winner, "debate": debate_results, "hist": hist}
    st.session_state['identifier_results'] = None
    st.session_state['run'] = False
    st.rerun()

# --- MANUAL ANALYSIS WORKFLOW ---
if st.session_state.get('run_manual', False):
    ticker = st.session_state['manual_ticker']
    horizon = st.session_state['horizon']
    if not ticker:
        st.error("Please enter a valid ticker symbol.")
        st.session_state['run_manual'] = False
    else:
        status = st.status(f"Analysing {ticker}...", expanded=True)
        try:
            status.write("📥 Fetching Market Data...")
            df = yf.download(ticker, period="1y", progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(ticker, axis=1, level=1)
                except: pass 
            
            score, metrics = calculate_technical_score(df, horizon)
            if not metrics:
                status.update(label=f"Error: Could not fetch valid data for {ticker}", state="error")
                st.stop()
                
            candidate_data = {"ticker": ticker}
            candidate_data.update(metrics)
            
            status.write("🧠 Gathering Context & Sentiment...")
            debate_results, hist = run_committee_meeting(client, gemini_key, candidate_data, horizon, allow_sell=True)
            
            status.update(label="Analysis Complete", state="complete", expanded=False)
            st.session_state['analysis_result'] = {"winner": candidate_data, "debate": debate_results, "hist": hist}
            st.session_state['identifier_results'] = None
            st.session_state['run_manual'] = False
            st.rerun()
        except Exception as e:
            status.update(label=f"Error analyzing {ticker}: {str(e)}", state="error")
            st.session_state['run_manual'] = False

# --- IDENTIFIER WORKFLOW ---
if st.session_state.get('run_identifier', False):
    topic = st.session_state['interest_topic']
    horizon = st.session_state['horizon']
    status = st.status(f"Surveying Market for: {topic}", expanded=True)
    
    try:
        # 1. Identify Tickers (Updated to use Gemini 3 Pro)
        status.write("🤖 Scanning Global Market for best exposure to theme (Powered by Gemini 3 Pro)...")
        # Passing gemini_key instead of 'client' (which was openai)
        identified_tickers = get_relevant_tickers(gemini_key, topic)
        
        if not identified_tickers:
            status.update(label=f"No relevant tickers found for {topic}", state="error")
            st.stop()
            
        status.write(f"🔎 Identified ({len(identified_tickers)}): {', '.join(identified_tickers)}")
        
        # 2. Rank Tickers (Technical + Sentiment)
        ranked_candidates = rank_identified_stocks(client, identified_tickers, horizon, status)
        
        if not ranked_candidates:
            status.update(label="No valid market data for identified stocks.", state="error")
            st.stop()
        
        # 3. Analyze Each Ticker (Deep Dive)
        results = []
        for i, candidate in enumerate(ranked_candidates):
            rank = i + 1
            tick = candidate['ticker']
            status.write(f"🧠 Deep Dive Analysis for #{rank}: {tick}...")
            
            debate, hist = run_committee_meeting(client, gemini_key, candidate, horizon, allow_sell=True, topic_interest=topic)
            results.append({"ticker": tick, "rank": rank, "debate": debate, "hist": hist})
            
        status.update(label="Survey Complete", state="complete", expanded=False)
        st.session_state['identifier_results'] = results
        st.session_state['analysis_result'] = None
        st.session_state['run_identifier'] = False
        st.rerun()
        
    except Exception as e:
        status.update(label=f"Identifier Error: {str(e)}", state="error")
        st.session_state['run_identifier'] = False

# --- DISPLAY RESULTS (SINGLE) ---
if st.session_state['analysis_result']:
    res = st.session_state['analysis_result']
    debate = res['debate']
    ticker = res['winner']['ticker']
    
    st.divider()
    st.header(f"Committee Transcript: {ticker}")
    
    with st.expander("🧠 Strategy Context (Thinking Process)"):
        st.text(debate.get('context_str', 'No Context Available'))
        
    col_news, col_sentiment = st.columns(2)
    with col_news:
        st.markdown("### 📰 The News Anchor (Facts)")
        st.info(debate['news_briefing'])
    with col_sentiment:
        st.markdown("### 🗣️ Social Sentiment Analysis")
        st.warning(debate['sentiment_report'])

    st.divider()
    col_bear, col_bull = st.columns(2)
    with col_bear:
        st.subheader("🐻 The Bear (Risks)")
        st.markdown(debate['bear']) 
    with col_bull:
        st.subheader("🐂 The Bull (Catalysts)")
        st.markdown(debate['bull']) 
        
    st.divider()
    st.subheader("👨‍⚖️ CIO Execution Order")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.success(debate['cio'])
    with c2:
        hist = res['hist']
        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(50).mean(), line=dict(color='orange'), name='SMA 50'))
        fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# --- DISPLAY RESULTS (IDENTIFIER LOOP) ---
if st.session_state.get('identifier_results'):
    st.divider()
    st.header(f"🎯 Market Survey Results")
    st.markdown("Stocks ranked by Technical Strength & Sentiment alignment.")
    
    results = st.session_state['identifier_results']
    
    for res in results:
        ticker = res['ticker']
        rank = res['rank']
        debate = res['debate']
        hist = res['hist']
        
        with st.expander(f"#{rank} {ticker} - Analysis Report", expanded=(rank==1)):
            st.subheader(f"#{rank} {ticker}")
            st.caption(f"Topic Match: {debate.get('topic', 'N/A')}")
            
            # Brief Summary
            c_bull, c_bear, c_cio = st.columns(3)
            with c_bull: st.info(f"**Bull Case:**\n{debate['bull'][:200]}...")
            with c_bear: st.error(f"**Bear Case:**\n{debate['bear'][:200]}...")
            with c_cio: st.success(f"**Verdict:**\n{debate['cio']}")
            
            # Chart
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(50).mean(), line=dict(color='orange'), name='SMA 50'))
            fig.update_layout(height=300, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Full Discussion")
            st.text(debate['news_briefing'])
            st.text(debate['sentiment_report'])
