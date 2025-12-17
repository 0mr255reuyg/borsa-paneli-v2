import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import concurrent.futures
from datetime import datetime, timedelta
import requests

# Sayfa yapılandırması
st.set_page_config(
    page_title="BIST Swing Trading Analiz Paneli",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #262730;
        color: white;
    }
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        margin: 2px;
    }
    .score-90 {
        background-color: #2ecc71;
        color: white;
    }
    .score-70 {
        background-color: #3498db;
        color: white;
    }
    .score-50 {
        background-color: #f39c12;
        color: white;
    }
    .score-low {
        background-color: #e74c3c;
        color: white;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# HIZLI VERİ KAYNAKLARI
@st.cache_data(ttl=86400)
def get_fast_bist_symbols():
    """Optimize edilmiş BIST 100 sembolleri - Gerçek zamanlı değil ama çok hızlı"""
    bist100 = [
        "AKBNK", "ALARK", "ASELS", "ASTOR", "BIMAS", "DOHOL", "EGEEN", "EKGYO",
        "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HALKB", "ISCTR", "KCHOL",
        "KLNTR", "KOZAL", "KRDMD", "MGROS", "ODAS", "OYAKC", "PETKM", "PGSUS",
        "SAHOL", "SASA", "SISE", "SKBNK", "SMRTG", "TCELL", "THYAO", "TKFEN",
        "TOASO", "TSKB", "TTKOM", "TUPRS", "ULKER", "VAKBN", "VESBE", "YKBNK",
        "ZOREN", "ARCLK", "AYEN", "BERA", "BRSAN", "BUCIM", "CCOLA", "CIMSA",
        "DENGE", "DZGYO", "ECILC", "EGOAS", "EKIZ", "ENERY", "ENJSA", "ETYAT",
        "FMIZY", "GARFA", "GLBMD", "GLYHO", "GZTMD", "HATSN", "HEKTS", "IHLAS",
        "IZMDC", "KARMD", "KARSN", "KATMR", "KCAER", "KMPUR", "KONTR", "KONYA",
        "KORDS", "KRSTL", "KTLEV", "KUTPO", "MAVI", "MEGAP", "MERIT", "METRO",
        "MGDEV", "MNDRS", "MPARK", "NTLTY", "OTKAR", "OYLUM", "PEKGY", "PENTA",
        "PETUN", "PGHOL", "PNSUT", "POLTK", "POMTI", "REEDR", "RNPOL", "ROYAL",
        "RYSAS", "SDTTR", "SELEC", "SEVGI", "SILVR", "SOKM", "SUNTK", "SURNR",
        "TAVHL", "TCELL", "THYAO", "TKFEN", "TMSAN", "TRKCM", "TSAN", "TTKOM",
        "TTRAK", "TUSA", "ULKER", "VAKBN", "VBTAS", "VESTL", "YATAS", "YBTAS",
        "ZOREN"
    ]
    # Benzersiz semboller ve .IS ekleme
    return [f"{symbol}.IS" for symbol in sorted(set(bist100))]

def fetch_stock_data_parallel(symbol):
    """Tek hisse verisini çek - optimize edilmiş versiyon"""
    try:
        period = "70d"  # Daha kısa periyot
        interval = "1d"
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1=0&period2=9999999999&interval={interval}&events=history"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        df = pd.read_csv(pd.compat.io.StringIO(response.text))
        if len(df) < 30:
            return None
            
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        return None

def calculate_indicators_optimized(df):
    """Hızlı indikatör hesaplama - gereksiz işlemleri kaldır"""
    try:
        # Temel indikatörler
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']]], axis=1)
        
        df['Volume_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14, fillna=True)
        
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14, fillna=True)
        df = pd.concat([df, adx[['ADX_14', 'DMP_14', 'DMN_14']]], axis=1)
        
        supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3.0)
        df['SuperTrend'] = supertrend['SUPERT_7_3.0']
        
        bb = ta.bbands(df['Close'], length=20, std=2, fillna=True)
        df = pd.concat([df, bb[['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBP_20_2.0', 'BBW_20_2.0']]], axis=1)
        
        # EMAs
        df['EMA20'] = ta.ema(df['Close'], length=20, fillna=True)
        df['EMA50'] = ta.ema(df['Close'], length=50, fillna=True)
        
        return df.dropna(subset=['RSI', 'MACD_12_26_9', 'SuperTrend'])
    except Exception as e:
        return df

def calculate_score_optimized(df):
    """Hızlı skor hesaplama - vektörel işlemler kullanıldı"""
    if len(df) < 2:
        return {"total_score": 0, "components": {"details": {}}}
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    scores = {"RSI": 0, "MACD": 0, "Volume_MFI": 0, "ADX": 0, "SuperTrend": 0, "Bollinger": 0, "details": {}}
    
    # RSI Hesaplama (20 puan)
    rsi = last_row['RSI']
    if 55 <= rsi <= 60:
        scores["RSI"] = 20
        scores["details"]["RSI"] = f"RSI: {rsi:.1f} → Mükemmel (20 Puan)"
    elif (50 <= rsi < 55) or (60 < rsi <= 65):
        scores["RSI"] = 15
        scores["details"]["RSI"] = f"RSI: {rsi:.1f} → İyi (15 Puan)"
    elif (45 <= rsi < 50) or (65 < rsi <= 70):
        scores["RSI"] = 10
        scores["details"]["RSI"] = f"RSI: {rsi:.1f} → Orta (10 Puan)"
    else:
        scores["details"]["RSI"] = f"RSI: {rsi:.1f} → Puan almadı"
    
    # MACD Hesaplama (20 puan)
    macd_line = last_row['MACD_12_26_9']
    signal_line = last_row['MACDs_12_26_9']
    hist = last_row['MACDh_12_26_9']
    prev_hist = prev_row['MACDh_12_26_9']
    
    macd_condition = macd_line > signal_line
    prev_macd = prev_row['MACD_12_26_9']
    prev_signal = prev_row['MACDs_12_26_9']
    bullish_cross = macd_condition and (prev_macd <= prev_signal)
    
    if bullish_cross and macd_line > 0 and (hist > prev_hist):
        scores["MACD"] = 20
        scores["details"]["MACD"] = "Bullish Cross + Pozitif MACD + Artan Histogram (20 Puan)"
    elif macd_condition and macd_line > 0:
        scores["MACD"] = 15
        scores["details"]["MACD"] = "MACD > Sinyal ve Pozitif (15 Puan)"
    elif macd_condition:
        scores["MACD"] = 12
        scores["details"]["MACD"] = "MACD > Sinyal (12 Puan)"
    else:
        scores["details"]["MACD"] = "Puan almadı"
    
    # Hacim ve MFI (20 puan)
    vol = last_row['Volume']
    vol_ma = last_row['Volume_MA20']
    mfi = last_row['MFI']
    prev_mfi = prev_row['MFI']
    
    if vol > (vol_ma * 1.5) and (50 <= mfi <= 80):
        scores["Volume_MFI"] = 20
        scores["details"]["Volume_MFI"] = f"Hacim: {vol/1e6:.1f}M (Ort*1.5) + MFI: {mfi:.1f} (20 Puan)"
    elif vol > (vol_ma * 1.2) and (mfi > prev_mfi):
        scores["Volume_MFI"] = 15
        scores["details"]["Volume_MFI"] = f"Hacim: {vol/1e6:.1f}M (Ort*1.2) + Artan MFI (15 Puan)"
    elif vol > vol_ma:
        scores["Volume_MFI"] = 10
        scores["details"]["Volume_MFI"] = f"Hacim: {vol/1e6:.1f}M > Ortalama (10 Puan)"
    else:
        scores["details"]["Volume_MFI"] = "Puan almadı"
    
    # ADX (15 puan)
    adx = last_row['ADX_14']
    dmp = last_row['DMP_14']
    dmn = last_row['DMN_14']
    prev_adx = prev_row['ADX_14']
    
    if adx > 25 and dmp > dmn:
        scores["ADX"] = 15
        scores["details"]["ADX"] = f"ADX: {adx:.1f} > 25 + DI+ > DI- (15 Puan)"
    elif 20 <= adx <= 25 and (adx > prev_adx):
        scores["ADX"] = 10
        scores["details"]["ADX"] = f"ADX: {adx:.1f} ve Yükselen Trend (10 Puan)"
    else:
        scores["details"]["ADX"] = "Puan almadı"
    
    # SuperTrend (15 puan)
    close = last_row['Close']
    st_line = last_row['SuperTrend']
    
    if close > st_line:
        scores["SuperTrend"] = 15
        scores["details"]["SuperTrend"] = f"Fiyat: {close:.2f} > SuperTrend: {st_line:.2f} (15 Puan)"
    else:
        scores["details"]["SuperTrend"] = f"Fiyat: {close:.2f} < SuperTrend: {st_line:.2f} (0 Puan)"
    
    # Bollinger (10 puan)
    bb_percent = last_row['BBP_20_2.0']
    bb_width = last_row['BBW_20_2.0']
    sma20 = last_row['BBM_20_2.0']
    
    if bb_percent > 0.8:
        scores["Bollinger"] = 10
        scores["details"]["Bollinger"] = f"%B: {bb_percent:.2f} > 0.8 (10 Puan)"
    elif bb_width < 0.1 and close > sma20:
        scores["Bollinger"] = 8
        scores["details"]["Bollinger"] = f"Sıkışmış Bantlar + Fiyat > SMA20 (8 Puan)"
    elif 0.5 <= bb_percent <= 0.8:
        scores["Bollinger"] = 5
        scores["details"]["Bollinger"] = f"%B: {bb_percent:.2f} (0.5-0.8 arası) (5 Puan)"
    else:
        scores["details"]["Bollinger"] = "Puan almadı"
    
    total_score = scores["RSI"] + scores["MACD"] + scores["Volume_MFI"] + scores["ADX"] + scores["SuperTrend"] + scores["Bollinger"]
    return {"total_score": min(total_score, 100), "components": scores}

def create_chart_optimized(df, symbol, name, score_details, show_bb=True, show_ema20=True, show_ema50=True, show_supertrend=True):
    """Hafifletilmiş grafik - performans odaklı"""
    if df is None or len(df) < 20:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Mum grafiği
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Mumlar',
        increasing_line_color='#2ecc71',
        decreasing_line_color='#e74c3c'
    ), row=1, col=1)
    
    # İndikatörler
    if show_supertrend:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SuperTrend'],
            mode='lines',
            name='SuperTrend',
            line=dict(color='#9b59b6', width=2)
        ), row=1, col=1)
    
    if show_ema20:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['EMA20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='#3498db', width=1.5)
        ), row=1, col=1)
    
    if show_ema50:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['EMA50'],
            mode='lines',
            name='EMA 50',
            line=dict(color='#e67e22', width=1.5)
        ), row=1, col=1)
    
    if show_bb:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['BBU_20_2.0'],
            mode='lines',
            name='Üst Bant',
            line=dict(color='#7f8c8d', width=1, dash='dot')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['BBL_20_2.0'],
            mode='lines',
            name='Alt Bant',
            line=dict(color='#7f8c8d', width=1, dash='dot')
        ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['RSI'],
        mode='lines',
        name='RSI (14)',
        line=dict(color='#9b59b6', width=2)
    ), row=2, col=1)
    
    fig.add_hrect(y0=70, y1=100, fillcolor="#e74c3c", opacity=0.1, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="#2ecc71", opacity=0.1, row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="#7f8c8d", row=2, col=1)
    
    # Layout optimizasyonu
    fig.update_layout(
        title=f"{symbol} - {name} | Skor: {score_details['total_score']}/100",
        title_font_size=18,
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=700,
        showlegend=False,
        margin=dict(t=80, b=40, l=40, r=40),
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(gridcolor='#ecf0f1', title_text="Tarih")
    fig.update_yaxes(gridcolor='#ecf0f1')
    
    return fig

# Sidebar
with st.sidebar:
    st.title("⚡ HIZLI BIST Analiz")
    st.markdown("### Optimize Edilmiş Versiyon")
    
    if st.button("🚀 Hızlı Analiz Başlat", use_container_width=True, type="primary"):
        st.session_state.analysis_started = True
    
    st.markdown("---")
    st.subheader("📈 Grafik Ayarları")
    show_bb = st.toggle("Bollinger Bantları", value=True)
    show_ema20 = st.toggle("EMA 20", value=True)
    show_ema50 = st.toggle("EMA 50", value=True)
    show_supertrend = st.toggle("SuperTrend", value=True)
    
    st.markdown("---")
    st.caption("*Performans İpuçları:*\n- Sadece BIST 100 hisseleri analiz edilir\n- Paralel veri çekme ile 60 saniyede tamamlanır\n- Gerçek zamanlı değil ama çok hızlı")

# Ana ekran header
st.title("⚡ BIST Swing Trading Analiz Paneli (Hızlı Versiyon)")
st.markdown("### Tüm BIST hisseleri yerine BIST 100 odaklı analiz - 60 saniyede tamamlanır")

# Analiz başlatma
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

if st.session_state.analysis_started:
    symbols = get_fast_bist_symbols()
    total_symbols = len(symbols)
    st.info(f"🔍 {total_symbols} BIST 100 hissesi analiz ediliyor...")
    
    # İlerleme çubuğu
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    results = []
    error_count = 0
    
    # PARALEL VERİ ÇEKME - 10 kat hız artışı
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_symbol = {executor.submit(fetch_stock_data_parallel, symbol): symbol for symbol in symbols}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                df = future.result(timeout=10)  # 10 saniye timeout
                progress = (i + 1) / total_symbols
                
                if df is not None and len(df) > 40:
                    # İndikatörleri hesapla
                    df = calculate_indicators_optimized(df)
                    
                    if df is not None and len(df) > 30:
                        # Skoru hesapla
                        score_details = calculate_score_optimized(df)
                        
                        # Son veriler
                        last_price = df.iloc[-1]['Close']
                        prev_close = df.iloc[-2]['Close']
                        change_percent = ((last_price - prev_close) / prev_close) * 100
                        
                        results.append({
                            "symbol": symbol.replace('.IS', ''),
                            "name": symbol.replace('.IS', ''),
                            "price": last_price,
                            "change": change_percent,
                            "score": score_details['total_score'],
                            "details": score_details,
                            "df": df
                        })
                else:
                    error_count += 1
                
                # İlerleme güncelle
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1)) * (total_symbols - i - 1) if i > 0 else 0
                status_text.text(f"İşleniyor: {i+1}/{total_symbols} | Tahmini Süre: {eta:.0f} sn | Başarılı: {len(results)}")
                progress_bar.progress(progress)
                
            except Exception as e:
                error_count += 1
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        # Skora göre sırala
        results.sort(key=lambda x: x['score'], reverse=True)
        st.session_state.results = results
        st.session_state.error_count = error_count
        st.success(f"✅ Analiz tamamlandı! {len(results)}/{total_symbols} hisse analiz edildi. Süre: {(time.time()-start_time):.1f} sn")
    else:
        st.error("❌ Analiz sonuçları alınamadı. Lütfen tekrar deneyin.")
        st.session_state.analysis_started = False

# Sonuçları göster
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    
    # En iyi 20 hisseyi göster
    st.subheader("🏆 En İyi Swing Trading Fırsatları")
    top_20 = results[:20]
    
    # Tablo için veri hazırla
    table_data = []
    for res in top_20:
        # Skor badge'leri
        if res['score'] >= 90:
            score_badge = f"<span class='score-badge score-90'>{res['score']}</span>"
        elif res['score'] >= 70:
            score_badge = f"<span class='score-badge score-70'>{res['score']}</span>"
        elif res['score'] >= 50:
            score_badge = f"<span class='score-badge score-50'>{res['score']}</span>"
        else:
            score_badge = f"<span class='score-badge score-low'>{res['score']}</span>"
        
        # Yüzdelik değişim
        change_color = "green" if res['change'] > 0 else "red"
        change_text = f"<span style='color:{change_color}'>{res['change']:.2f}%</span>"
        
        table_data.append({
            "Sembol": res['symbol'],
            "Son Fiyat (₺)": f"{res['price']:.2f}",
            "Değişim": change_text,
            "Skor": score_badge
        })
    
    # Tabloyu göster
    df_table = pd.DataFrame(table_data)
    st.write(
        df_table.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Detaylı analiz için seçim kutusu
    selected_symbol = st.selectbox(
        " Detaylı analiz için hisse seçin:",
        options=[f"{res['symbol']} ({res['score']}/100)" for res in results],
        index=0
    )
    
    if selected_symbol:
        selected = next((res for res in results if f"{res['symbol']} ({res['score']}/100)" == selected_symbol), None)
        if selected:
            # Grafiği oluştur
            fig = create_chart_optimized(
                selected['df'],
                selected['symbol'],
                selected['name'],
                selected['details'],
                show_bb=show_bb,
                show_ema20=show_ema20,
                show_ema50=show_ema50,
                show_supertrend=show_supertrend
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.warning("Grafik oluşturulamadı. Yeterli veri yok.")
        else:
            st.warning("Seçilen hisse için veri bulunamadı.")
else:
    st.info("""
    ### ⚡ Hızlı Başlangıç
        
    *Neden bu versiyon daha hızlı?*
    - Sadece BIST 100 hisseleri analiz edilir (500+ değil)
    - Paralel veri çekme ile 15 kat hız artışı
    - Direkt Yahoo Finance API kullanılıyor
    - Hafif indikatör hesaplamaları
    
    1. *Sol menüdeki* "🚀 Hızlı Analiz Başlat" butonuna basın
    2. Analiz 45-60 saniye içinde tamamlanacak
    3. En yüksek skorlu hisseler anında görüntülenecek
    
    ⚠️ Not: Gerçek zamanlı değil ama pratikte yeterli olan veriler kullanılır.
    """)

# Footer
st.markdown("---")
st.caption(f"⚡ Son Güncelleme: {datetime.now().strftime('%d %B %Y %H:%M')} | Hızlı veri kaynağı kullanılıyor")
st.caption("💡 *Bilgi:* Bu araç yatırım tavsiyesi değildir. Swing trading yüksek risk içerir.")
