import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import numpy as np
import time
from datetime import datetime, timedelta

# Sayfa yapılandırması
st.set_page_config(
    page_title="BIST Swing Trading Analiz Paneli",
    page_icon="📈",
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
        background-color: #1f77b4;
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

# Veri çekme fonksiyonları
@st.cache_data(ttl=86400)  # 24 saat cache
def get_bist_symbols():
    """BIST tüm hisselerin sembollerini GitHub'dan çek"""
    try:
        url = "https://raw.githubusercontent.com/urazakgul/bist-symbols/master/bist_stock_symbols.csv"
        response = requests.get(url)
        symbols = pd.read_csv(pd.compat.io.StringIO(response.text))['symbol'].tolist()
        return [f"{symbol}.IS" for symbol in symbols if symbol not in ['GARAN', 'YKBNK', 'ISCTR']]  # Sorunlu sembolleri filtrele
    except Exception as e:
        st.error(f"Sembol listesi çekilirken hata oluştu: {str(e)}")
        # Yedek sembol listesi
        return ["AKBNK.IS", "THYAO.IS", "FROTO.IS", "ASELS.IS", "SISE.IS", "KCHOL.IS", "TUPRS.IS", "EREGL.IS", "PGSUS.IS", "SAHOL.IS"]

@st.cache_data(ttl=7200)  # 2 saat cache
def fetch_stock_data(symbol, period="3mo"):
    """Tek bir hissenin verisini yfinance ile çek"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval="1d")
        if len(df) < 30:  # Yeterli veri yoksa
            return None
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Veri çekilirken hata oluştu ({symbol}): {str(e)}")
        return None

def calculate_indicators(df):
    """Tüm teknik indikatörleri hesapla"""
    try:
        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # Hacim ve MFI
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        
        # ADX
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df = pd.concat([df, adx], axis=1)
        
        # SuperTrend
        supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3.0)
        df['SuperTrend'] = supertrend['SUPERT_7_3.0']
        
        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=20, std=2)
        df = pd.concat([df, bb], axis=1)
        df['BB_PERCENT'] = (df['Close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        
        # EMAs
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        
        return df.dropna()
    except Exception as e:
        print(f"İndikatör hesaplanırken hata: {str(e)}")
        return df

def calculate_score(df):
    """Swing Skorunu hesapla"""
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else None
    
    scores = {
        "RSI": 0,
        "MACD": 0,
        "Volume_MFI": 0,
        "ADX": 0,
        "SuperTrend": 0,
        "Bollinger": 0,
        "details": {}
    }
    
    # 1. RSI (14) - Maks 20 Puan
    rsi = last_row['RSI']
    if 55 <= rsi <= 60:
        scores["RSI"] = 20
        scores["details"]["RSI"] = f"RSI: {rsi:.2f} → Mükemmel (20 Puan)"
    elif (50 <= rsi < 55) or (60 < rsi <= 65):
        scores["RSI"] = 15
        scores["details"]["RSI"] = f"RSI: {rsi:.2f} → İyi (15 Puan)"
    elif (45 <= rsi < 50) or (65 < rsi <= 70):
        scores["RSI"] = 10
        scores["details"]["RSI"] = f"RSI: {rsi:.2f} → Orta (10 Puan)"
    else:
        scores["details"]["RSI"] = f"RSI: {rsi:.2f} → Puan almadı"
    
    # 2. MACD (12,26,9) - Maks 20 Puan
    macd_line = last_row['MACD_12_26_9']
    signal_line = last_row['MACDs_12_26_9']
    hist = last_row['MACDh_12_26_9']
    prev_hist = prev_row['MACDh_12_26_9'] if prev_row is not None else 0
    
    macd_condition = macd_line > signal_line
    bullish_cross = False
    if prev_row is not None:
        prev_macd = prev_row['MACD_12_26_9']
        prev_signal = prev_row['MACDs_12_26_9']
        bullish_cross = (macd_line > signal_line) and (prev_macd <= prev_signal)
    
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
    
    # 3. Hacim ve MFI (14) - Maks 20 Puan
    vol = last_row['Volume']
    vol_ma = last_row['Volume_MA20']
    mfi = last_row['MFI']
    prev_mfi = prev_row['MFI'] if prev_row is not None else 0
    
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
    
    # 4. ADX (14) - Maks 15 Puan
    adx = last_row['ADX_14']
    dmp = last_row['DMP_14']
    dmn = last_row['DMN_14']
    prev_adx = prev_row['ADX_14'] if prev_row is not None else 0
    
    if adx > 25 and dmp > dmn:
        scores["ADX"] = 15
        scores["details"]["ADX"] = f"ADX: {adx:.1f} > 25 + DI+ > DI- (15 Puan)"
    elif 20 <= adx <= 25 and (adx > prev_adx):
        scores["ADX"] = 10
        scores["details"]["ADX"] = f"ADX: {adx:.1f} ve Yükselen Trend (10 Puan)"
    else:
        scores["details"]["ADX"] = "Puan almadı"
    
    # 5. SuperTrend (7,3) - Maks 15 Puan
    close = last_row['Close']
    st_line = last_row['SuperTrend']
    
    if close > st_line:
        scores["SuperTrend"] = 15
        scores["details"]["SuperTrend"] = f"Fiyat: {close:.2f} > SuperTrend: {st_line:.2f} (15 Puan)"
    else:
        scores["details"]["SuperTrend"] = f"Fiyat: {close:.2f} < SuperTrend: {st_line:.2f} (0 Puan)"
    
    # 6. Bollinger Bantları (20,2) - Maks 10 Puan
    bb_percent = last_row['BB_PERCENT']
    bb_width = last_row['BBW_20_2.0']
    sma20 = last_row['BBM_20_2.0']
    
    if bb_percent > 0.8:
        scores["Bollinger"] = 10
        scores["details"]["Bollinger"] = f"%B: {bb_percent:.2f} > 0.8 (10 Puan)"
    elif bb_width < 0.1 and close > sma20:  # Sıkışma ve SMA20 üzerinde
        scores["Bollinger"] = 8
        scores["details"]["Bollinger"] = f"Sıkışmış Bantlar + Fiyat > SMA20 (8 Puan)"
    elif 0.5 <= bb_percent <= 0.8:
        scores["Bollinger"] = 5
        scores["details"]["Bollinger"] = f"%B: {bb_percent:.2f} (0.5-0.8 arası) (5 Puan)"
    else:
        scores["details"]["Bollinger"] = "Puan almadı"
    
    # Toplam skor
    total_score = (
        scores["RSI"] + scores["MACD"] + scores["Volume_MFI"] +
        scores["ADX"] + scores["SuperTrend"] + scores["Bollinger"]
    )
    
    return {
        "total_score": min(total_score, 100),
        "components": scores
    }

def create_chart(df, symbol, name, score_details, show_bb=True, show_ema20=True, show_ema50=True, show_supertrend=True):
    """Detaylı finansal grafik oluştur"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('', 'Hacim', 'RSI')
    )
    
    # Mum grafiği
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Mumlar',
            increasing_line_color='#2ecc71',
            decreasing_line_color='#e74c3c'
        ),
        row=1, col=1
    )
    
    # İndikatörler
    if show_supertrend:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['SuperTrend'],
                mode='lines',
                name='SuperTrend (7,3)',
                line=dict(color='#9b59b6', width=2)
            ), row=1, col=1
        )
    
    if show_ema20:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['EMA20'],
                mode='lines',
                name='EMA 20',
                line=dict(color='#3498db', width=1.5)
            ), row=1, col=1
        )
    
    if show_ema50:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['EMA50'],
                mode='lines',
                name='EMA 50',
                line=dict(color='#e67e22', width=1.5)
            ), row=1, col=1
        )
    
    if show_bb:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['BBU_20_2.0'],
                mode='lines',
                name='Üst Bant',
                line=dict(color='#34495e', width=1, dash='dot')
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['BBL_20_2.0'],
                mode='lines',
                name='Alt Bant',
                line=dict(color='#34495e', width=1, dash='dot')
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['BBM_20_2.0'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#1abc9c', width=1.5)
            ), row=1, col=1
        )
    
    # Hacim
    colors = ['#2ecc71' if df['Close'].iloc[i] > df['Open'].iloc[i] else '#e74c3c' for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df['Date'], y=df['Volume'],
            name='Hacim',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=df['Date'], y=df['RSI'],
            mode='lines',
            name='RSI (14)',
            line=dict(color='#9b59b6', width=2)
        ),
        row=3, col=1
    )
    fig.add_hrect(y0=70, y1=100, fillcolor="#e74c3c", opacity=0.1, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="#2ecc71", opacity=0.1, row=3, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="#7f8c8d", row=3, col=1)
    
    # Grafik düzenlemeleri
    fig.update_layout(
        title=f"{symbol} - {name} | Toplam Skor: {score_details['total_score']}/100",
        title_font_size=20,
        title_x=0.5,
        title_y=0.98,
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100, b=50)
    )
    
    fig.update_xaxes(rangeslider_visible=False, gridcolor='#ecf0f1', title_text="Tarih")
    fig.update_yaxes(gridcolor='#ecf0f1')
    
    # Skor detaylarını annotasyon olarak ekle
    annotations = []
    y_pos = 0.95
    for component, detail in score_details['components']['details'].items():
        color = '#27ae60' if any(kw in detail for kw in ['Mükemmel', '20 Puan', '15 Puan', '12 Puan']) else '#e67e22'
        if '0 Puan' in detail or 'alınmadı' in detail:
            color = '#e74c3c'
        
        annotations.append(dict(
            xref='paper', yref='paper',
            x=1.02, y=y_pos,
            text=f"<b>{component}:</b> {detail}",
            showarrow=False,
            align='left',
            font=dict(color=color, size=12),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=color,
            borderwidth=1,
            borderpad=4
        ))
        y_pos -= 0.04
    
    fig.update_layout(annotations=annotations)
    
    return fig

# Sidebar
with st.sidebar:
    st.title("📊 BIST Swing Trading Analiz")
    st.markdown("### Teknik Analiz Parametreleri")
    
    if st.button("🎯 Analizi Başlat", use_container_width=True, type="primary"):
        st.session_state.analysis_started = True
    
    st.markdown("---")
    st.subheader("📈 Grafik Ayarları")
    show_bb = st.toggle("Bollinger Bantları", value=True)
    show_ema20 = st.toggle("EMA 20", value=True)
    show_ema50 = st.toggle("EMA 50", value=True)
    show_supertrend = st.toggle("SuperTrend", value=True)
    
    st.markdown("---")
    st.caption("💡 *Skorlama Sistemi*\n- RSI (14): 20 Puan\n- MACD (12,26,9): 20 Puan\n- Hacim & MFI: 20 Puan\n- ADX (14): 15 Puan\n- SuperTrend: 15 Puan\n- Bollinger: 10 Puan")

# Ana ekran
st.title("🚀 BIST Swing Trading Analiz Paneli")
st.markdown("### Günlük swing trading fırsatlarını keşfedin")

# Analiz başlatma
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

if st.session_state.analysis_started:
    symbols = get_bist_symbols()
    total_symbols = len(symbols)
    
    # İlerleme çubuğu
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    error_count = 0
    
    # Her sembol için analiz yap
    for i, symbol in enumerate(symbols):
        try:
            status_text.text(f"Analiz ediliyor: {symbol} ({i+1}/{total_symbols})")
            progress_bar.progress((i + 1) / total_symbols)
            
            # Veriyi çek ve indikatörleri hesapla
            df = fetch_stock_data(symbol)
            if df is None or len(df) < 50:
                error_count += 1
                continue
            
            df = calculate_indicators(df)
            if df is None or len(df) < 30:
                error_count += 1
                continue
            
            # Skoru hesapla
            score_details = calculate_score(df)
            
            # Son verileri al
            last_price = df.iloc[-1]['Close']
            prev_close = df.iloc[-2]['Close'] if len(df) > 1 else last_price
            change_percent = ((last_price - prev_close) / prev_close) * 100
            
            # Hisse adını al
            try:
                stock_info = yf.Ticker(symbol).info
                name = stock_info.get('shortName', symbol.replace('.IS', ''))
            except:
                name = symbol.replace('.IS', '')
            
            results.append({
                "symbol": symbol.replace('.IS', ''),
                "name": name,
                "price": last_price,
                "change": change_percent,
                "score": score_details['total_score'],
                "details": score_details,
                "df": df  # Grafiği çizmek için veriyi sakla
            })
            
            # Rate limit koruması
            time.sleep(0.1)
            
        except Exception as e:
            error_count += 1
            print(f"Hata: {symbol} - {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        # Skora göre sırala
        results.sort(key=lambda x: x['score'], reverse=True)
        st.session_state.results = results
        st.session_state.error_count = error_count
        st.success(f"✅ Analiz tamamlandı! {len(results)}/{total_symbols} hisse başarıyla analiz edildi. ({error_count} hata)")
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
            "İsim": res['name'],
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
        options=[f"{res['symbol']} - {res['name']}" for res in results],
        index=0
    )
    
    if selected_symbol:
        selected = next((res for res in results if f"{res['symbol']} - {res['name']}" == selected_symbol), None)
        if selected:
            # Grafiği oluştur
            fig = create_chart(
                selected['df'],
                selected['symbol'],
                selected['name'],
                selected['details'],
                show_bb=show_bb,
                show_ema20=show_ema20,
                show_ema50=show_ema50,
                show_supertrend=show_supertrend
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Skor detayları
            with st.expander("📊 Skor Detayları ve Açıklamalar"):
                st.subheader(f"{selected['symbol']} - {selected['name']} Skor Analizi")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 📈 Teknik İndikatörler")
                    for component, detail in selected['details']['components']['details'].items():
                        if "20 Puan" in detail or "15 Puan" in detail:
                            st.success(detail)
                        elif "12 Puan" in detail or "10 Puan" in detail or "8 Puan" in detail:
                            st.warning(detail)
                        else:
                            st.error(detail)
                
                with col2:
                    st.markdown("#### 💡 Swing Trading Stratejisi")
                    strategy_points = [
                        "🔹 *Giriş Noktası:* SuperTrend üzerinde ve RSI 55-60 aralığında",
                        "🔹 *Hacim Onayı:* Son 20 günlük ortalamanın %50 üzerinde hacim",
                        "🔹 *Trend Gücü:* ADX > 25 ve DI+ > DI-",
                        "🔹 *Çıkış Stratejisi:* RSI > 70 veya fiyat SuperTrend altına düşerse",
                        "🔹 *Stop Loss:* SuperTrend çizgisinin %2 altı"
                    ]
                    for point in strategy_points:
                        st.markdown(point)
                    
                    st.info(f"*Toplam Skor:* {selected['score']}/100")
                    if selected['score'] >= 90:
                        st.success("⭐ *Mükemmel Swing Fırsatı!* Tüm kriterler olumlu.")
                    elif selected['score'] >= 70:
                        st.warning("✅ *İyi Fırsat* - Temkinli pozisyon alınabilir.")
                    elif selected['score'] >= 50:
                        st.error("⚠️ *Dikkatli Olun* - Sadece tecrübeli yatırımcılar için.")
                    else:
                        st.error("❌ *Önerilmez* - Yeterli teknik sinyal yok.")
        else:
            st.warning("Seçilen hisse için veri bulunamadı.")
else:
    st.info("""
    ### 🚀 Başlamak İçin
        
    1. *Sol menüdeki* "🎯 Analizi Başlat" butonuna basın
    2. Tüm BIST hisseleri otomatik olarak taranacak
    3. En yüksek skorlu hisseler tabloda görüntülenecek
    4. Detaylı analiz için bir hisse seçin
        
    ⏱️ *Not:* İlk analiz 5-10 dakika sürebilir. Sonraki analizler cache sayesinde daha hızlı olacaktır.
    """)

# Footer
st.markdown("---")
st.caption(f"🔄 Son Güncelleme: {datetime.now().strftime('%d %B %Y %H:%M')} | Veriler Yahoo Finance ve BIST'ten alınmıştır")
st.caption("💡 *Bilgi:* Bu araç yatırım tavsiyesi değildir. Swing trading yüksek risk içerir. Lütfen kendi araştırma ve risk yönetiminizi yapın.")
