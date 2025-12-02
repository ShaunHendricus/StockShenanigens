import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
import google.generativeai as genai
import requests
import io
import time
import datetime
import feedparser

# --- CONFIGURATION ---
st.set_page_config(page_title="Institutional Committee", layout="wide", page_icon="⚖️")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# --- UNIVERSE GENERATOR (STRICT GITHUB ONLY) ---
@st.cache_data(ttl=3600*24)
def get_sp500_tickers():
    """
    Fetches S&P 500 tickers strictly from GitHub.
    No hardcoded backups. No Wikipedia fallbacks.
    """
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        # Ensure we get the ticker symbol column
        return df['Symbol'].tolist()
    except Exception as e:
        st.error(f"CRITICAL ERROR: Could not fetch S&P 500 list from GitHub.\nReason: {e}")
        return []

# --- NEWS ENGINE (Google RSS) ---
def get_google_news(ticker, limit=5):
    """
    Fetches official Google News RSS feed.
    Filters strictly for articles < 30 days old.
    """
    try:
        encoded_ticker = requests.utils.quote(f"{ticker} stock news")
        rss_url = f"https://news.google.com/rss/search?q={encoded_ticker}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(rss_url)
        
        valid_articles = []
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
        
        for entry in feed.entries:
            if hasattr(entry, 'published_parsed'):
                pub_date = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed))
                if pub_date > cutoff_date:
                    clean_date = pub_date.strftime('%Y-%m-%d')
                    valid_articles.append(f"- {entry.title} ({entry.source.title}, {clean_date})")
        
        return valid_articles[:limit]
        
    except Exception as e:
        return [f"News Error: {str(e)}"]

# --- STAGE 1: MATH FILTER (WEIGHTED SCORECARD FIX) ---
# --- STAGE 1: MATH FILTER (WEIGHTED SCORECARD FIX) ---
def run_quantitative_dragnet(tickers, horizon, progress_bar):
    if not tickers:
        return []

    progress_bar.write(f"📥 Downloading data for {len(tickers)} companies...")
    
    try:
        # Download ALL data at once
        data = yf.download(tickers, period="1y", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        st.error(f"YFinance Download Error: {e}")
        return []

    candidates = []
    progress_bar.write("🧮 Grading stocks using Weighted Scorecard (No strict cutoffs)...")

    for ticker in tickers:
        try:
            # Handle Single vs Multi Index
            if len(tickers) == 1:
                df = data
            else:
                df = data[ticker]
            
            df = df.dropna()
            if len(df) < 150: continue 

            close = df['Close']
            curr_price = close.iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_200 = close.rolling(200).mean().iloc[-1]
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.00001)
            rsi = 100 - (100 / (1 + (gain/loss))).iloc[-1]

            # --- THE SCORECARD LOGIC ---
            score = 0

            # 1. Trend Score (Up to 40pts)
            if curr_price > sma_200: score += 20
            if curr_price > sma_50: score += 20

            # 2. Strategy Score (Up to 60pts)
            if horizon == "Short": 
                # Dip Buy: We want Low RSI.
                # Note: We do NOT delete if Price < SMA 200. We just score it.
                if rsi < 30: score += 60       # Extreme Oversold
                elif rsi < 40: score += 40     # Oversold
                elif rsi < 50: score += 20     # Neutral
                elif rsi > 70: score -= 20     # Overbought penalty
                
                # CONTRARIAN BONUS: If price crashed (< SMA 200) AND RSI is extreme (< 25)
                # This catches "Dead Cat Bounces" or Reversals
                if curr_price < sma_200 and rsi < 25:
                    score += 50

            elif horizon == "Medium": 
                # Momentum: Price > 50 SMA & Healthy RSI
                if curr_price > sma_50:
                    if 50 < rsi < 70: score += 60 # Perfect Zone
                    elif rsi > 70: score += 40    # Strong but hot
                    elif rsi < 40: score -= 10    # Weak
            
            elif horizon == "Long": 
                # Value: Golden Cross & Pullback
                if sma_50 > sma_200: 
                    score += 40
                    # Bonus for entry near SMA 50
                    dist = abs((curr_price - sma_50)/sma_50 * 100)
                    if dist < 5: score += 20 # Perfect entry

            # Keep if score is decent
            if score > 40:
                candidates.append({
                    "ticker": ticker,
                    "math_score": score,
                    "price": curr_price,
                    "rsi": rsi,
                    "sma50": sma_50,
                    "sma200": sma_200
                })
        except: continue
            
    # Return ALL candidates that passed the score threshold
    # No more hard limit of 60.
    candidates.sort(key=lambda x: x['math_score'], reverse=True)
    return candidates

# --- STAGE 2: JUNIOR ANALYST ---
def run_junior_analyst_batch(client, candidates, progress_bar):
    progress_bar.write(f"🕵️ Junior Analyst scanning headlines for Top {len(candidates)}...")
    
    scored_candidates = []
    # Process in chunks or simple loop
    for cand in candidates:
        ticker = cand['ticker']
        
        # Get News
        news_items = get_google_news(ticker, limit=3)
        headlines = " | ".join(news_items) if news_items else "No recent news."
        
        # Optimization: If Math Score is extremely high (>100), skip AI to save time/cost
        # and assume good sentiment, OR prioritize it.
        # Here we run AI on all 60 for thoroughness.
        
        prompt = f"Rate sentiment of {ticker} based on headlines: '{headlines}'. Return score 0-10 only. 0=Bad, 10=Good."
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5
            )
            sentiment_score = float(response.choices[0].message.content.strip())
        except: sentiment_score = 5.0

        cand['final_score'] = cand['math_score'] + (sentiment_score * 5)
        cand['headlines'] = headlines
        scored_candidates.append(cand)
        
    scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return scored_candidates[0]

# --- STAGE 3: THE COMMITTEE MEETING ---
def run_committee_meeting(openai_client, gemini_key, winner_data, horizon):
    ticker = winner_data['ticker']
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    
    # 1. Fetch Financials
    try:
        info = stock.info
        sector = info.get('sector', 'Unknown')
        pe_ratio = info.get('trailingPE', 'N/A')
        fwd_pe = info.get('forwardPE', 'N/A')
        beta = info.get('beta', 'N/A')
    except: 
        sector, pe_ratio, fwd_pe, beta = "Unknown", "N/A", "N/A", "N/A"

    # 2. Fetch News (Using Robust RSS Engine)
    news_list = get_google_news(ticker, limit=5)
    full_news_text = "\n".join(news_list) if news_list else "No recent news found."

    context_data = f"""
    Ticker: {ticker}
    Sector: {sector}
    Price: ${winner_data['price']:.2f}
    RSI: {winner_data['rsi']:.1f}
    Trailing P/E: {pe_ratio}
    Forward P/E: {fwd_pe}
    Beta: {beta}
    Price vs 50 SMA: {"Above" if winner_data['price'] > winner_data['sma50'] else "Below"}
    """

    # --- AGENT 1: NEWS ANCHOR ---
    news_prompt = f"""
    You are a Financial News Anchor.
    Extract purely factual material events from these headlines.
    
    HEADLINES:
    {full_news_text}
    
    TASK:
    List the top 3 events (Earnings, Legal, Products) with DATES.
    NO Opinions. Just Facts.
    """
    news_briefing = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": news_prompt}],
        temperature=0.0
    ).choices[0].message.content

    # --- AGENT 2: THE BEAR ---
    bear_argument = "Gemini API Error."
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            bear_prompt = f"""
            Role: Risk Analyst (Bear).
            Analyze {ticker} based on these FACTS.
            
            [TECHNICALS]: {context_data}
            [NEWS]: {news_briefing}
            
            Task: Identify 3 risks (Valuation, Overbought RSI, Bad News).
            Tone: Skeptical, Warning.
            """
            bear_argument = model.generate_content(bear_prompt).text
        except Exception as e:
            bear_argument = f"Risk Analysis Unavailable: {e}"

    # --- AGENT 3: THE BULL ---
    bull_prompt = f"""
    Role: Growth Strategist (Bull).
    Analyze {ticker} based on these FACTS.
    
    [TECHNICALS]: {context_data}
    [NEWS]: {news_briefing}
    
    Task: Identify 3 catalysts (Momentum, Value, Good News).
    Tone: Professional, Opportunistic.
    """
    bull_argument = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": bull_prompt}],
        temperature=0.1
    ).choices[0].message.content

# --- AGENT 4: THE CIO ---
    cio_prompt = f"""
    Role: Chief Investment Officer.
    
    [NEWS]: {news_briefing}
    [BEAR]: {bear_argument}
    [BULL]: {bull_argument}
    
    [DATA]: Price ${winner_data['price']:.2f}, RSI {winner_data['rsi']:.1f}
    
    Task: Final Execution Order.
    1. Verdict (BUY / WAIT).
    2. Limit Price.
    3. Stop Loss (Calculated from volatility).
    4. Rationale (Synthesis of News + Math).
    
    Tone: Directive. No fluff.
    """

    # UPDATED: Integration of GPT-5.1 using the new responses API
    try:
        response = openai_client.responses.create(
            model="gpt-5.1",
            input=cio_prompt
        )
        final_verdict = response.output_text
    except Exception as e:
        final_verdict = f"CIO Error: {str(e)}"
    
    return {
        "raw_news": full_news_text, 
        "news_briefing": news_briefing,
        "bear": bear_argument,
        "bull": bull_argument,
        "cio": final_verdict
    }, hist

# --- UI LOGIC ---
with st.sidebar:
    st.header("🔑 Credentials")
    openai_key = st.text_input("OpenAI API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")

st.title("⚖️ Institutional Committee: S&P 500")

if not openai_key:
    st.warning("Please enter OpenAI Key.")
    st.stop()

client = OpenAI(api_key=openai_key)
col1, col2, col3 = st.columns(3)

if col1.button("🚀 Short Term"):
    st.session_state['horizon'] = "Short"
    st.session_state['run'] = True
if col2.button("📈 Medium Term"):
    st.session_state['horizon'] = "Medium"
    st.session_state['run'] = True
if col3.button("💰 Long Term"):
    st.session_state['horizon'] = "Long"
    st.session_state['run'] = True

if st.session_state.get('run', False):
    horizon = st.session_state['horizon']
    status = st.status("The Committee is in session...", expanded=True)
    
    # FETCH UNIVERSE
    status.write("🌎 Accessing S&P 500 Database (GitHub)...")
    tickers = get_sp500_tickers()
    
    if not tickers:
        status.update(label="Critical Error: S&P 500 List not found.", state="error")
        st.stop()
    
    # STAGE 1
    top_60 = run_quantitative_dragnet(tickers, horizon, status)
    if not top_60:
        st.error("No stocks met the criteria.")
        st.stop()
        
    # STAGE 2
    status.write(f"🤖 Junior Analyst scanning Top {len(top_60)} for News Sentiment...")
    winner = run_junior_analyst_batch(client, top_60, status)
    status.write(f"🔔 Candidate Selected: **{winner['ticker']}**")
    
    # STAGE 3
    status.write("📰 News Anchor is briefing the committee...")
    debate_results, hist = run_committee_meeting(client, gemini_key, winner, horizon)
    
    status.update(label="Meeting Adjourned", state="complete", expanded=False)
    
    st.session_state['analysis_result'] = {"winner": winner, "debate": debate_results, "hist": hist}
    st.session_state['run'] = False
    st.rerun()

if st.session_state['analysis_result']:
    res = st.session_state['analysis_result']
    debate = res['debate']
    ticker = res['winner']['ticker']
    
    st.divider()
    st.header(f"Committee Transcript: {ticker}")

    # 1. NEWS SECTION
    st.markdown("### 📰 Impartial News Wire")
    st.info(debate['news_briefing']) 
    
    with st.expander("See Raw News Data (Debugging)"):
        st.text(debate['raw_news'])
    
    st.divider()

    # 2. DEBATE
    col_bear, col_bull = st.columns(2)
    with col_bear:
        st.subheader("🐻 The Bear (Risks)")
        st.markdown(debate['bear']) 
    with col_bull:
        st.subheader("🐂 The Bull (Catalysts)")
        st.markdown(debate['bull']) 
        
    st.divider()
    
    # 3. VERDICT
    st.subheader("👨‍⚖️ CIO Execution Order")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.success(debate['cio'])
    with c2:
        hist = res['hist']
        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(50).mean(), line=dict(color='orange'), name='SMA 50'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(), line=dict(color='blue'), name='SMA 200'))
        fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
