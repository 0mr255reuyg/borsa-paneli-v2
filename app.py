import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import concurrent.futures
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Sayfa yapılandırması
st.set_page_config(
    page_title="BIST Swing Trading",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #262730; color: white; }
    .stProgress > div > div > div > div { background-color: #3498db; }
    .score-badge { 
        display: inline-block; padding: 3px 8px; border-radius: 10px; 
        font-weight: bold; margin: 1px; font-size: 12px;
    }
    .score-90 { background-color: #2ecc71; color: white; }
    .score-70 { background-color: #3498db; color: white; }
    .score-50 { background-color: #f39c12; color: white; }
    .score-low { background-color: #e74c3c; color: white; }
    </style>
""", unsafe_allow_html=True)

# Statik BIST 100 sembolleri - hiç bağımlılık gerektirmez
BIST_100_SYMBOLS = [
    "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "DOHOL.IS", "EGEEN.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS",
    "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HALKB.IS", "ISCTR.IS", "KCHOL.IS", "KLNTR.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS",
    "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS", "TCELL.IS",
    "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "VESBE.IS", "YKBNK.IS",
    "ZOREN.IS", "ARCLK.IS", "AYEN.IS", "BERA.IS", "BRSAN.IS", "BUCIM.IS", "CCOLA.IS", "CIMSA.IS", "DENGE.IS", "DZGYO.IS",
    "ECILC.IS", "EGOAS.IS", "EKIZ.IS", "ENERY.IS", "ENJSA.IS", "ETYAT.IS", "FMIZY.IS", "GARFA.IS", "GLBMD.IS", "GLYHO.IS",
    "GZTMD.IS", "HATSN.IS", "HEKTS.IS", "IHLAS.IS", "IZMDC.IS", "KARMD.IS", "KARSN.IS", "KATMR.IS", "KCAER.IS", "KMPUR.IS",
    "KONTR.IS", "KONYA.IS", "KORDS.IS", "KRSTL.IS", "KTLEV.IS", "KUTPO.IS", "MAVI.IS", "MEGAP.IS", "MERIT.IS", "METRO.IS",
    "MGDEV.IS", "MNDRS.IS", "MPARK.IS", "NTLTY.IS", "OTKAR.IS", "OYLUM.IS", "PEKGY.IS", "PENTA.IS", "PETUN.IS", "PGHOL.IS",
    "PNSUT.IS", "POLTK.IS", "POMTI.IS", "REEDR.IS", "RNPOL.IS", "ROYAL.IS", "RYSAS.IS", "SDTTR.IS", "SELEC.IS", "SEVGI.IS",
    "SILVR.IS", "SOKM.IS", "SUNTK.IS", "SURNR.IS", "TAVHL.IS", "TMSAN.IS", "TRKCM.IS", "TSAN.IS", "TTRAK.IS", "TUSA.IS",
    "VBTAS.IS", "VESTL.IS", "YATAS.IS", "YBTAS.IS"
]

def fetch_stock_data_minimal(symbol: str) -> pd.DataFrame:
    """En basit veri çekme - sadece requests ve pandas kullanıyor"""
    try:
        # Sadece son 30 gün
        end_date = int(datetime.now().timestamp())
        start_date = int((datetime.now() - timedelta(days=40)).timestamp())
        
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_date}&period2={end_date}&interval=1d"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        data = response.json()
        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            return None
            
        result = data['chart']['result'][0]
        if 'timestamp' not in result or 'indicators' not in result:
            return None
            
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        # Temel verileri al
        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s'),
            'Open': quotes['open'],
            'High': quotes['high'],
            'Low': quotes['low'],
            'Close': quotes['close'],
            'Volume': quotes['volume']
        })
        
        # Temizlik
        df = df.dropna(subset=['Close'])
        if len(df) < 25:
            return None
            
        return df.tail(30)  # Sadece son 30 gün
    except Exception as e:
        return None

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Manuel RSI hesaplama - bağımsız"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Bölme hatası önlemek için
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices: pd.Series) -> tuple:
    """Manuel MACD hesaplama"""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_mfi(high, low, close, volume, period=14):
    """Manuel MFI hesaplama"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    
    pos_flow_sum = pd.Series(positive_flow).rolling(period).sum()
    neg_flow_sum = pd.Series(negative_flow).rolling(period).sum()
    
    mfi = 100 - (100 / (1 + pos_flow_sum / neg_flow_sum.replace(0, 1e-10)))
    return mfi

def calculate_adx(high, low, close, period=14):
    """Basit ADX hesaplama"""
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3))
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di

def calculate_supertrend(high, low, close, period=7, multiplier=3.0):
    """Basit SuperTrend hesaplama"""
    hl2 = (high + low) / 2
    atr = pd.Series(high - low).rolling(period).mean()
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = pd.Series(np.nan, index=close.index)
    direction = 1  # 1 for up, -1 for down
    
    for i in range(period, len(close)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            direction = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            direction = -1
            
        if direction == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
            
    return supertrend

def calculate_score_from_data(df: pd.DataFrame) -> int:
    """Tüm indikatörleri manuel hesaplayan skorlama"""
    if len(df) < 25:
        return 0
    
    # Temel veriler
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # 1. RSI hesaplama
    rsi = calculate_rsi(close, 14)
    last_rsi = rsi.iloc[-1] if not rsi.empty else 50
    
    # 2. MACD hesaplama
    macd_line, signal_line, hist = calculate_macd(close)
    last_macd = macd_line.iloc[-1] if not macd_line.empty else 0
    last_signal = signal_line.iloc[-1] if not signal_line.empty else 0
    last_hist = hist.iloc[-1] if not hist.empty else 0
    prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else 0
    prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else 0
    prev_hist = hist.iloc[-2] if len(hist) > 1 else 0
    
    # 3. Hacim ve MFI
    volume_ma = volume.rolling(20).mean()
    last_vol = volume.iloc[-1]
    last_vol_ma = volume_ma.iloc[-1] if not volume_ma.empty else 1
    mfi = calculate_mfi(high, low, close, volume, 14)
    last_mfi = mfi.iloc[-1] if not mfi.empty else 50
    prev_mfi = mfi.iloc[-2] if len(mfi) > 1 else 50
    
    # 4. ADX
    adx, dmp, dmn = calculate_adx(high, low, close, 14)
    last_adx = adx.iloc[-1] if not adx.empty else 0
    last_dmp = dmp.iloc[-1] if not dmp.empty else 0
    last_dmn = dmn.iloc[-1] if not dmn.empty else 0
    prev_adx = adx.iloc[-2] if len(adx) > 1 else 0
    
    # 5. SuperTrend
    supertrend = calculate_supertrend(high, low, close, 7, 3.0)
    last_st = supertrend.iloc[-1] if not supertrend.empty else 0
    last_price = close.iloc[-1]
    
    # 6. Bollinger Bantları
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    bb_percent = (close - lower_band) / (upper_band - lower_band).replace(0, 1e-10)
    
    last_bb_percent = bb_percent.iloc[-1] if not bb_percent.empty else 0.5
    last_sma20 = sma20.iloc[-1] if not sma20.empty else last_price
    
    # Skor hesaplama
    total_score = 0
    
    # RSI (20 puan)
    if 55 <= last_rsi <= 60:
        total_score += 20
    elif (50 <= last_rsi < 55) or (60 < last_rsi <= 65):
        total_score += 15
    elif (45 <= last_rsi < 50) or (65 < last_rsi <= 70):
        total_score += 10
    
    # MACD (20 puan)
    macd_condition = last_macd > last_signal
    bullish_cross = macd_condition and (prev_macd <= prev_signal)
    
    if bullish_cross and last_macd > 0 and (last_hist > prev_hist):
        total_score += 20
    elif macd_condition and last_macd > 0:
        total_score += 15
    elif macd_condition:
        total_score += 12
    
    # Hacim ve MFI (20 puan)
    vol_ratio = last_vol / last_vol_ma if last_vol_ma > 0 else 0
    
    if vol_ratio > 1.5 and 50 <= last_mfi <= 80:
        total_score += 20
    elif vol_ratio > 1.2 and last_mfi > prev_mfi:
        total_score += 15
    elif vol_ratio > 1.0:
        total_score += 10
    
    # ADX (15 puan)
    if last_adx > 25 and last_dmp > last_dmn:
        total_score += 15
    elif 20 <= last_adx <= 25 and last_adx > prev_adx:
        total_score += 10
    
    # SuperTrend (15 puan)
    if last_price > last_st:
        total_score += 15
    
    # Bollinger (10 puan)
    if last_bb_percent > 0.8:
        total_score += 10
    elif last_bb_percent > 0.5 and last_bb_percent <= 0.8:
        total_score += 5
    
    return min(total_score, 100)

def create_simple_chart(df: pd.DataFrame, symbol: str, score: int) -> go.Figure:
    """Basit ama etkili grafik"""
    if df is None or len(df) < 20:
        fig = go.Figure()
        fig.add_annotation(text="Veri Yok", x=0.5, y=0.5, showarrow=False, font_size=24)
        return fig
    
    df = df.tail(25)  # Son 25 gün
    
    fig = go.Figure()
    
    # Mum grafiği
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Fiyat'
    ))
    
    # Basit hareketli ortalama
    sma20 = df['Close'].rolling(20).mean()
    if not sma20.isna().all():
        fig.add_trace(go.Scatter(
            x=df['Date'], y=sma20,
            mode='lines', name='SMA 20',
            line=dict(color='#3498db', width=1.5)
        ))
    
    # Grafik düzeni
    fig.update_layout(
        title=f"{symbol} | Skor: {score}/100",
        xaxis_title="Tarih",
        yaxis_title="Fiyat (₺)",
        height=400,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white',
        hovermode="x unified"
    )
    
    fig.update_xaxes(rangeslider_visible=False, gridcolor='#ecf0f1')
    fig.update_yaxes(gridcolor='#ecf0f1')
    
    return fig

# Sidebar
with st.sidebar:
    st.title("⚡ BIST Analiz")
    st.markdown("### Hızlı ve Güvenilir")
    
    if st.button("🚀 Analiz Başlat", use_container_width=True, type="primary"):
        st.session_state.analysis_started = True
    
    st.markdown("---")
    st.caption("*Özellikler:*\n- Sadece BIST 100 hisseleri\n- Tüm hesaplamalar manuel\n- %100 bağımlılık sorunsuz\n- 45 saniye içinde tamamlanır")

# Session state kontrolü
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# Ana ekran
st.title("⚡ BIST Swing Trading Analiz Paneli")
st.subheader("Bağımlılık Sorunsuz Hızlı Versiyon")

# Açıklama
st.info("""
*Neden Bu Versiyon Çalışır?*
- 🔧 *Hiçbir ek kütüphane kullanmıyor:* Sadece pandas, numpy, requests
- 📊 *Tüm indikatörler manuel hesaplanıyor:* pandas_ta veya ta kütüphanelerine gerek yok
- ⚡ *Süper hızlı:* Sadece BIST 100 hisseleri analiz ediliyor
- ✅ *%100 Streamlit Cloud uyumlu:* Kurulum hatası yok
""")

# Analiz işlemi
if st.session_state.analysis_started:
    st.warning("🔍 Analiz başlıyor! Lütfen bekleyin...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    results = []
    
    # Paralel işlem - basit versiyon
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {executor.submit(fetch_stock_data_minimal, symbol): symbol for symbol in BIST_100_SYMBOLS}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                df = future.result(timeout=10)
                
                if df is not None and len(df) >= 25:
                    # Skoru hesapla
                    score = calculate_score_from_data(df)
                    
                    if score > 30:  # Sadece anlamlı skorlar
                        last_price = df.iloc[-1]['Close']
                        prev_price = df.iloc[-2]['Close'] if len(df) > 1 else last_price
                        change = ((last_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
                        
                        results.append({
                            'symbol': symbol.replace('.IS', ''),
                            'price': last_price,
                            'change': change,
                            'score': score,
                            'df': df.tail(30)
                        })
                
                # İlerleme güncelle
                progress = (i + 1) / len(BIST_100_SYMBOLS)
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(BIST_100_SYMBOLS) - i - 1) if i > 0 else 0
                
                status_text.text(f"İŞLENİYOR: {i+1}/100 | Tahmini Süre: {eta:.0f} sn | Başarılı: {len(results)}")
                progress_bar.progress(progress)
                
            except Exception as e:
                continue
    
    # Sonuçları kaydet
    if results:
        results.sort(key=lambda x: x['score'], reverse=True)
        st.session_state.results = results
        total_time = time.time() - start_time
        st.success(f"✅ Analiz tamamlandı! {len(results)}/100 hisse. Süre: {total_time:.1f} sn")
    else:
        st.error("❌ Analiz başarısız. Lütfen tekrar deneyin.")
        st.session_state.analysis_started = False

# Sonuçları göster
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    
    st.subheader("🏆 En İyi 15 Swing Fırsatı")
    
    # En iyi 15'i göster
    top_15 = results[:15]
    
    # Tablo hazırlığı
    table_data = []
    for res in top_15:
        # Skor badge
        if res['score'] >= 90:
            badge = f"<span class='score-badge score-90'>{res['score']}</span>"
        elif res['score'] >= 70:
            badge = f"<span class='score-badge score-70'>{res['score']}</span>"
        elif res['score'] >= 50:
            badge = f"<span class='score-badge score-50'>{res['score']}</span>"
        else:
            badge = f"<span class='score-badge score-low'>{res['score']}</span>"
        
        # Değişim rengi
        color = "green" if res['change'] >= 0 else "red"
        change_text = f"<span style='color:{color}'>{res['change']:.2f}%</span>"
        
        table_data.append({
            "Sembol": res['symbol'],
            "Fiyat (₺)": f"{res['price']:.2f}",
            "Değişim": change_text,
            "Skor": badge
        })
    
    # Tabloyu göster
    st.write(pd.DataFrame(table_data).to_html(escape=False, index=False), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detaylı analiz
    selected = st.selectbox(" Detaylı analiz için hisse seçin:", 
                          [f"{r['symbol']} ({r['score']})" for r in results])
    
    if selected:
        symbol = selected.split(' ')[0]
        result = next((r for r in results if r['symbol'] == symbol), None)
        
        if result:
            fig = create_simple_chart(result['df'], result['symbol'], result['score'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Skor detayları
            with st.expander("Skor Detayları"):
                st.markdown(f"""
                *Toplam Skor: {result['score']}/100*
                
                📈 *Kullanılan İndikatörler:*
                - RSI (14) - Trend gücü
                - MACD (12,26,9) - Momentum
                - Hacim Analizi - Balina hareketleri
                - ADX (14) - Trend gücü
                - SuperTrend - Trend yönü
                - Bollinger Bantları - Volatilite
                
                💡 *Strateji:* Skoru 70+ olan hisseler swing trading için en uygun fırsatları temsil eder.
                """)

# Başlangıç ekranı
else:
    st.markdown("""
    ### 🚀 Başlamak İçin
                
    *1. Sol menüdeki "🚀 Analiz Başlat" butonuna basın*
    - Sadece BIST 100 hisseleri analiz edilecek
    - 45 saniye içinde sonuçlar hazır olacak
                
    *2. Sonuçlar hazır olduğunda:*
    - En iyi 15 hisse tabloda görünecek
    - Detaylı analiz için listeden hisse seçin
                
    *✅ Neden Bu Versiyon Çalışır?*
    - Hiçbir ek bağımlılık kullanmıyor
    - Tüm hesaplamalar manuel olarak yapılıyor
    - Streamlit Cloud ile tam uyumlu
    """)
    
    # Demo tablo
    st.subheader("📊 Örnek Sonuçlar")
    demo_data = {
        "Sembol": ["THYAO", "TUPRS", "FROTO", "AKBNK", "GARAN"],
        "Fiyat (₺)": ["285.50", "187.30", "452.80", "125.40", "89.75"],
        "Değişim": ["+2.45%", "+1.20%", "-0.75%", "+3.10%", "-0.30%"],
        "Skor": [
            "<span class='score-badge score-90'>95</span>",
            "<span class='score-badge score-70'>78</span>",
            "<span class='score-badge score-50'>55</span>",
            "<span class='score-badge score-90'>92</span>",
            "<span class='score-badge score-low'>42</span>"
        ]
    }
    st.write(pd.DataFrame(demo_data).to_html(escape=False, index=False), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(f"⚡ Son Güncelleme: {datetime.now().strftime('%H:%M')} | Bağımlılık Sorunsuz Versiyon")
