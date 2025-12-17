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

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BIST Swing Trader Pro v2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS ile görünümü güzelleştirelim
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-weight: bold;
    }
    .score-box {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ ÇEKME VE LİSTELEME FONKSİYONLARI
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # 1 saatlik önbellek
def get_bist_symbols():
    """
    Geniş kapsamlı BIST hisse listesini döndürür.
    Buraya internetten çekilen bir liste veya manuel geniş liste eklenebilir.
    Şimdilik popüler ve hacimli 100+ hisseyi ve fazlasını kapsayacak şekilde ayarlıyoruz.
    """
    # Örnek geniş liste (BIST 100 + Bazı Yan Tahtalar)
    # Not: Gerçek tüm liste için github raw url kullanılabilir, şimdilik manuel geniş liste.
    symbols = [
        "THYAO", "ASELS", "SISE", "KCHOL", "AKBNK", "YKBNK", "GARAN", "ISCTR", "EREGL", "TUPRS",
        "SAHOL", "BIMAS", "FROTO", "TOASO", "TCELL", "TTKOM", "PETKM", "KOZAL", "KOZAA", "IPEKE",
        "KRDMD", "SASA", "HEKTS", "ASTOR", "KONTR", "SMRTG", "EGEEN", "GUBRF", "EKGYO", "ODAS",
        "ZOREN", "VESTL", "ARCLK", "PGSUS", "ENKAI", "ALARK", "TKFEN", "TAVHL", "MGROS", "SOKM",
        "AEFES", "AGHOL", "AKSA", "AKSEN", "ALGYO", "ALKIM", "AYDEM", "BAGFS", "BERA", "BRYAT",
        "BUCIM", "CCOLA", "CEMTS", "CIMSA", "DOAS", "DOHOL", "ECILC", "EUREN", "GENIL", "GESAN",
        "GLYHO", "GOZDE", "GWIND", "HALKB", "ISDMR", "ISGYO", "ISMEN", "JANTS", "KARSN", "KMPUR",
        "KORDS", "MAVI", "NTHOL", "OYAKC", "PENTA", "QUAGR", "RTALB", "SKBNK", "SNGYO", "TATGD",
        "TUKAS", "ULKER", "VAKBN", "VESBE", "YYLGD", "YEOTK", "CANTE", "EUPWR", "CVKMD", "KOPOL",
        "ONCSM", "SDTTR", "TNZTP", "GOKNR", "AKFYE", "BIGCH"
        # Buraya dilediğin kadar ekleyebilirsin, sonuna .IS ekleyeceğiz.
    ]
    return [f"{s}.IS" for s in symbols]

def fetch_data(symbol):
    """Yahoo Finance'den veri çeker."""
    try:
        # Son 6 aylık veri yeterli
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if len(df) < 50:  # Yetersiz veri varsa atla
            return None
        return df
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# 3. İNDİKATÖR VE PUANLAMA MANTIĞI (SENİN 100 PUANLIK SİSTEMİN)
# -----------------------------------------------------------------------------
def calculate_indicators_and_score(df):
    if df is None or df.empty:
        return 0, {}, df

    # --- İndikatör Hesaplamaları ---
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    # Sütun isimlerini düzeltme (bazen farklı gelebilir)
    macd_col = 'MACD_12_26_9'
    signal_col = 'MACDs_12_26_9'
    hist_col = 'MACDh_12_26_9'

    # MFI & Hacim Ortalaması
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['Vol_MA'] = df['Volume'].rolling(20).mean()

    # ADX
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df = pd.concat([df, adx], axis=1)
    adx_col = 'ADX_14'
    dmp_col = 'DMP_14'
    dmn_col = 'DMN_14'

    # SuperTrend
    st_data = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
    df = pd.concat([df, st_data], axis=1)
    st_col = 'SUPERT_10_3.0' # SuperTrend çizgi değeri

    # Bollinger
    bb = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1)
    bbl_col = 'BBL_20_2.0'
    bbu_col = 'BBU_20_2.0'
    bbm_col = 'BBM_20_2.0' # SMA
    df['B_Percent'] = (df['Close'] - df[bbl_col]) / (df[bbu_col] - df[bbl_col])

    # EMA'lar (Grafik için)
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['EMA50'] = ta.ema(df['Close'], length=50)

    # Son satırı al (En güncel veri)
    current = df.iloc[-1]
    prev = df.iloc[-2]

    # --- PUANLAMA MANTIĞI (100 PUAN) ---
    score = 0
    details = []

    # 1. RSI (20 Puan)
    rsi_val = current['RSI']
    if 55 <= rsi_val <= 60:
        score += 20
        details.append("RSI Mükemmel (55-60)")
    elif (50 <= rsi_val < 55) or (60 < rsi_val <= 65):
        score += 15
        details.append("RSI İyi (50-55 veya 60-65)")
    elif (45 <= rsi_val < 50) or (65 < rsi_val <= 70):
        score += 10
        details.append("RSI Orta")
    
    # 2. MACD (20 Puan)
    # Bullish cross: MACD > Sinyal VE önceki gün MACD < Sinyal
    macd_val = current[macd_col]
    sig_val = current[signal_col]
    bullish_cross = (macd_val > sig_val) and (prev[macd_col] <= prev[signal_col])
    
    if bullish_cross and macd_val > 0: # Pozitif bölgede kesişim
        score += 20
        details.append("MACD: Pozitif Kesişim (Güçlü)")
    elif macd_val > sig_val and macd_val > 0: # Pozitif ve yukarıda
        score += 15
        details.append("MACD: Pozitif Trend")
    elif macd_val > sig_val and macd_val < 0: # Negatif bölgede yukarı kesmiş
        score += 12
        details.append("MACD: Dipten Dönüş")

    # 3. Hacim ve MFI (20 Puan)
    vol_cond = current['Volume'] > (current['Vol_MA'] * 1.5)
    mfi_val = current['MFI']
    
    if vol_cond and (50 <= mfi_val <= 80):
        score += 20
        details.append("Hacim Patlaması & MFI İdeal")
    elif (current['Volume'] > current['Vol_MA'] * 1.2) and (mfi_val > prev['MFI']):
        score += 15
        details.append("Hacim Artışı & MFI Yükseliyor")
    elif current['Volume'] > current['Vol_MA']:
        score += 10
        details.append("Hacim Ortalamanın Üstünde")

    # 4. ADX (15 Puan)
    adx_val = current[adx_col]
    if adx_val > 25 and current[dmp_col] > current[dmn_col]:
        score += 15
        details.append("ADX: Güçlü Trend (>25)")
    elif 20 <= adx_val <= 25 and adx_val > prev[adx_col]:
        score += 10
        details.append("ADX: Trend Güçleniyor")

    # 5. SuperTrend (15 Puan)
    # Fiyat > SuperTrend
    if current['Close'] > current[st_col]:
        score += 15
        details.append("SuperTrend: AL (Fiyat Üstte)")

    # 6. Bollinger (10 Puan)
    b_pct = current['B_Percent']
    if b_pct > 0.8:
        score += 10
        details.append("Bollinger: Üst Banda Yakın (>0.8)")
    elif 0.5 <= b_pct <= 0.8:
        score += 5
        details.append("Bollinger: Üst Yarıda")

    return score, details, df

# -----------------------------------------------------------------------------
# 4. ARAYÜZ (UI) TASARIMI
# -----------------------------------------------------------------------------
# -- Sidebar --
st.sidebar.title("Kanka'nın Borsa Paneli v2")
st.sidebar.info("Bu modül 'Pro Mod' mantığıyla, 500+ hisseyi sırayla tarayıp puanlar.")

# Hisse Listesi Seçimi (İleride tümünü seçebilirsin)
# symbol_list = get_bist_symbols() # Otomatik liste
# Manuel ekleme opsiyonu da olsun
manual_input = st.sidebar.text_area("Hisse Ekle (Virgülle ayır)", "GUBRF, SMRTG, EGEEN")
if manual_input:
    extras = [x.strip().upper() + ".IS" for x in manual_input.split(",")]
else:
    extras = []

# -- Ana Ekran --
st.title("🚀 BIST Swing Trader Pro - Fırsat Avcısı")
st.markdown("Hacmi patlayan, indikatörleri 'AL' veren gizli hisseleri bul.")

# Başlat Butonu (Sitenin donmaması için kilit nokta)
if st.sidebar.button("🎯 TARAMAYI BAŞLAT", type="primary"):
    
    # Listeyi hazırla
    full_list = get_bist_symbols() + extras
    full_list = list(set(full_list)) # Tekrar edenleri sil
    
    st.write(f"Toplam {len(full_list)} hisse taranıyor... Bu işlem biraz sürebilir, kahveni al bekle ☕")
    
    progress_bar = st.progress(0)
    status_txt = st.empty()
    
    results = []
    
    # Döngü Başlıyor
    for i, sembol in enumerate(full_list):
        # Progress güncelle
        progress = (i + 1) / len(full_list)
        progress_bar.progress(progress)
        status_txt.text(f"İnceleniyor: {sembol}")
        
        try:
            # 1. Veri Çek
            df = fetch_data(sembol)
            if df is None: continue
            
            # 2. Puanla
            score, details, processed_df = calculate_indicators_and_score(df)
            
            # 3. Sonuçları Kaydet
            last_price = processed_df['Close'].iloc[-1]
            change = ((processed_df['Close'].iloc[-1] - processed_df['Close'].iloc[-2]) / processed_df['Close'].iloc[-2]) * 100
            
            # Sadece puanı yüksek olanları listeye al (Örn: 50 üstü)
            if score >= 50: 
                results.append({
                    "Sembol": sembol.replace(".IS", ""),
                    "Fiyat": f"{last_price:.2f} ₺",
                    "Değişim": f"%{change:.2f}",
                    "Puan": score,
                    "Nedenler": ", ".join(details)
                })
            
            # Rate limit yememek için minik bekleme
            time.sleep(0.05)
            
        except Exception as e:
            # Hata olsa bile devam et, sistemi durdurma
            continue

    progress_bar.empty()
    status_txt.success("✅ Tarama Tamamlandı!")
    
    # -- SONUÇLARI GÖSTER --
    if results:
        # Puan sırasına göre diz (En yüksek en üstte)
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="Puan", ascending=False).reset_index(drop=True)
        
        st.subheader("🏆 EN YÜKSEK PUANLI HİSSELER")
        st.dataframe(df_results, use_container_width=True)
        
        # En iyisini detaylı göster
        best_stock = df_results.iloc[0]["Sembol"]
        st.info(f"En Gözde Hisse: *{best_stock}* - Puan: {df_results.iloc[0]['Puan']}")
        
    else:
        st.warning("Kriterlere uyan hisse bulunamadı veya veri çekilemedi.")

else:
    st.write("👈 Taramayı başlatmak için soldaki butona bas kanka.")
    st.write("Sistem şu an hazırda bekliyor...")
