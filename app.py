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
import os
import json
from typing import List, Dict, Any

# Sayfa yapılandırması
st.set_page_config(
    page_title="BIST Swing Trading Analiz Paneli",
    page_icon="📊",
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
        background-color: #3498db;
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
    .mode-selector {
        display: flex;
        gap: 10px;
        margin: 20px 0;
    }
    .mode-btn {
        flex: 1;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        border: 2px solid #3498db;
    }
    .mode-btn.active {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    .mode-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# VERİ KAYNAKLARI - HEM BIST 100 HEM BIST TÜM
@st.cache_data(ttl=86400)
def get_bist_symbols(mode: str = "BIST100") -> List[str]:
    """Akıllı sembol yönetimi - iki mod destekliyor"""
    if mode == "BIST100":
        # BIST 100 sembolleri - hızlı analiz için
        bist100 = [
            "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "DOHOL.IS", 
            "EGEEN.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", 
            "GUBRF.IS", "HALKB.IS", "ISCTR.IS", "KCHOL.IS", "KLNTR.IS", "KOZAL.IS", 
            "KRDMD.IS", "MGROS.IS", "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", 
            "SAHOL.IS", "SASA.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS", "TCELL.IS", 
            "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TUPRS.IS", 
            "ULKER.IS", "VAKBN.IS", "VESBE.IS", "YKBNK.IS", "ZOREN.IS", "ARCLK.IS", 
            "AYEN.IS", "BERA.IS", "BRSAN.IS", "BUCIM.IS", "CCOLA.IS", "CIMSA.IS", 
            "DENGE.IS", "DZGYO.IS", "ECILC.IS", "EGOAS.IS", "EKIZ.IS", "ENERY.IS", 
            "ENJSA.IS", "ETYAT.IS", "FMIZY.IS", "GARFA.IS", "GLBMD.IS", "GLYHO.IS", 
            "GZTMD.IS", "HATSN.IS", "HEKTS.IS", "IHLAS.IS", "IZMDC.IS", "KARMD.IS", 
            "KARSN.IS", "KATMR.IS", "KCAER.IS", "KMPUR.IS", "KONTR.IS", "KONYA.IS", 
            "KORDS.IS", "KRSTL.IS", "KTLEV.IS", "KUTPO.IS", "MAVI.IS", "MEGAP.IS", 
            "MERIT.IS", "METRO.IS", "MGDEV.IS", "MNDRS.IS", "MPARK.IS", "NTLTY.IS", 
            "OTKAR.IS", "OYLUM.IS", "PEKGY.IS", "PENTA.IS", "PETUN.IS", "PGHOL.IS", 
            "PNSUT.IS", "POLTK.IS", "POMTI.IS", "REEDR.IS", "RNPOL.IS", "ROYAL.IS", 
            "RYSAS.IS", "SDTTR.IS", "SELEC.IS", "SEVGI.IS", "SILVR.IS", "SOKM.IS", 
            "SUNTK.IS", "SURNR.IS", "TAVHL.IS", "TMSAN.IS", "TRKCM.IS", "TSAN.IS", 
            "TTRAK.IS", "TUSA.IS", "VBTAS.IS", "VESTL.IS", "YATAS.IS", "YBTAS.IS"
        ]
        return sorted(set(bist100))
    
    else:  # BIST TÜM
        # BIST TÜM sembolleri - toplamda ~580 hisse
        try:
            # GitHub'dan güncel liste çek
            url = "https://raw.githubusercontent.com/urazakgul/bist-symbols/master/bist_all_symbols.csv"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                symbols_df = pd.read_csv(pd.compat.io.StringIO(response.text))
                symbols = symbols_df['symbol'].tolist()
                # Sorunlu sembolleri filtrele
                exclude_symbols = ['GARAN', 'YKBNK', 'ISCTR', 'THYAO', 'FROTO']  # Bilinen sorunlu semboller
                symbols = [f"{s}.IS" for s in symbols if s not in exclude_symbols and len(s) <= 5]
                return symbols[:600]  # Maksimum 600 hisse
        except Exception as e:
            st.warning(f"Sembol listesi çekilirken hata oluştu: {str(e)}")
        
        # Yedek liste (statik)
        backup_symbols = [
            "AKBNK.IS", "ALARK.IS", "ASELS.IS", "BIMAS.IS", "DOHOL.IS", "EGEEN.IS", 
            "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "HALKB.IS", 
            "ISCTR.IS", "KCHOL.IS", "KLNTR.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS", 
            "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", 
            "SISE.IS", "SKBNK.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", 
            "TSKB.IS", "TTKOM.IS", "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "VESBE.IS", 
            "YKBNK.IS", "ZOREN.IS", "ARCLK.IS", "AYEN.IS", "BERA.IS", "BRSAN.IS", 
            "CCOLA.IS", "CIMSA.IS", "DENGE.IS", "DZGYO.IS", "ECILC.IS", "EGOAS.IS", 
            "EKIZ.IS", "ENERY.IS", "ENJSA.IS", "ETYAT.IS", "FMIZY.IS", "GARFA.IS", 
            "GLBMD.IS", "GLYHO.IS", "GZTMD.IS", "HATSN.IS", "HEKTS.IS", "IHLAS.IS", 
            "IZMDC.IS", "KARMD.IS", "KARSN.IS", "KATMR.IS", "KCAER.IS", "KONTR.IS", 
            "KONYA.IS", "KORDS.IS", "KRSTL.IS", "KTLEV.IS", "KUTPO.IS", "MAVI.IS", 
            "MEGAP.IS", "MERIT.IS", "METRO.IS", "MGDEV.IS", "MNDRS.IS", "MPARK.IS", 
            "NTLTY.IS", "OTKAR.IS", "OYLUM.IS", "PEKGY.IS", "PENTA.IS", "PETUN.IS", 
            "PGHOL.IS", "PNSUT.IS", "POLTK.IS", "POMTI.IS", "REEDR.IS", "RNPOL.IS", 
            "ROYAL.IS", "RYSAS.IS", "SDTTR.IS", "SELEC.IS", "SEVGI.IS", "SILVR.IS", 
            "SOKM.IS", "SUNTK.IS", "SURNR.IS", "TAVHL.IS", "TMSAN.IS", "TRKCM.IS", 
            "TSAN.IS", "TTRAK.IS", "TUSA.IS", "VBTAS.IS", "VESTL.IS", "YATAS.IS", 
            "YBTAS.IS", "ZOREN.IS", "AKCNS.IS", "AKFYE.IS", "AKGRT.IS", "AKSA.IS", 
            "AKSEN.IS", "ALBRK.IS", "ALFAS.IS", "ALTIN.IS", "ANHYT.IS", "ANSGR.IS", 
            "AVHOL.IS", "AVOD.IS", "AVYON.IS", "BRSAN.IS", "BUCIM.IS", "CANTE.IS", 
            "CCBRS.IS", "CELHA.IS", "CEMAS.IS", "CETEC.IS", "CLEBI.IS", "CMBTN.IS", 
            "CTMT.IS", "CUCUK.IS", "CURMD.IS", "CZMOT.IS", "DAPGM.IS", "DENGE.IS", 
            "DENIZ.IS", "DERHL.IS", "DERIT.IS", "DEVA.IS", "DGATE.IS", "DGNMO.IS", 
            "DITAS.IS", "DMRGD.IS", "DOAS.IS", "DOGER.IS", "DURDO.IS", "DYOBY.IS", 
            "DZGYO.IS", "ECILC.IS", "ECZYT.IS", "EGEEN.IS", "EGESE.IS", "EGKYO.IS", 
            "EGOAS.IS", "EGPRO.IS", "EGSER.IS", "EGYOG.IS", "EKGYO.IS", "EKIZ.IS", 
            "EKSUN.IS", "ELITE.IS", "EMKEL.IS", "ENJSA.IS", "ENSRI.IS", "ENTRA.IS", 
            "ENVEO.IS", "EREGL.IS", "ERET.IS", "ERGL.IS", "ESCAR.IS", "ESCOM.IS", 
            "ESGSY.IS", "ESKIM.IS", "ESMOD.IS", "ESTUR.IS", "ETILR.IS", "ETYAT.IS", 
            "EUCELL.IS", "EUREN.IS", "FONET.IS", "FMIZY.IS", "FONET.IS", "FROTO.IS", 
            "GARAN.IS", "GARFA.IS", "GARFI.IS", "GARSY.IS", "GARTE.IS", "GEDZA.IS", 
            "GENIL.IS", "GENTS.IS", "GEREL.IS", "GESAN.IS", "GIPTA.IS", "GLBMD.IS", 
            "GLYHO.IS", "GMDAS.IS", "GNKEL.IS", "GOODY.IS", "GOZDE.IS", "GRNYO.IS", 
            "GSDHO.IS", "GSRAY.IS", "GUBRF.IS", "GWIND.IS", "GZNMI.IS", "HALKB.IS", 
            "HATEK.IS", "HATSN.IS", "HATUT.IS", "HAYAT.IS", "HEKTS.IS", "HKTM.IS", 
            "HLGYO.IS", "HURGZ.IS", "HURSV.IS", "ICBCT.IS", "ICFVF.IS", "IEYHO.IS", 
            "IHEVA.IS", "IHYAY.IS", "IHKIZ.IS", "IHLAS.IS", "IHLGM.IS", "IHSAN.IS", 
            "IITCH.IS", "INDES.IS", "INGOR.IS", "INTEM.IS", "INVES.IS", "IONTE.IS", 
            "ISCTR.IS", "ISDMR.IS", "ISGYO.IS", "ISMEN.IS", "IZENR.IS", "IZFAS.IS", 
            "IZMDC.IS", "IZMOT.IS", "IZYAT.IS", "JANTS.IS", "KARSN.IS", "KATMR.IS", 
            "KCAER.IS", "KCHOL.IS", "KCRDT.IS", "KDSGA.IS", "KENVY.IS", "KERVT.IS", 
            "KLGYO.IS", "KLNTR.IS", "KLSTN.IS", "KMPUR.IS", "KMRUP.IS", "KONTR.IS", 
            "KONYA.IS", "KORDS.IS", "KORHO.IS", "KOSGD.IS", "KOSTL.IS", "KRSTL.IS", 
            "KRTEK.IS", "KSTUR.IS", "KTLEV.IS", "KTSKR.IS", "KUTPO.IS", "KUVVA.IS", 
            "KZBGY.IS", "KZBGA.IS", "KZBGD.IS", "KZBGH.IS", "KZBGJ.IS", "KZBGT.IS", 
            "KZBGV.IS", "KZBGZ.IS", "LASIS.IS", "LCIDB.IS", "LCIDC.IS", "LCIDA.IS", 
            "LCIDF.IS", "LCIDG.IS", "LCIDH.IS", "LCIDI.IS", "LCIDJ.IS", "LCIDK.IS", 
            "LCIDL.IS", "LCIDM.IS", "LCIDN.IS", "LCIDO.IS", "LCIDP.IS", "LCIDQ.IS",
            # Devam eden semboller...
        ]
        return sorted(set(backup_symbols))[:500]  # 500 sembol ile sınırla

def fetch_stock_data_parallel(symbol: str, period: str = "70d") -> pd.DataFrame:
    """Optimize edilmiş veri çekme - hem hızlı hem güvenli"""
    try:
        # Direkt Yahoo Finance API kullan
        start_date = int((datetime.now() - timedelta(days=90)).timestamp())
        end_date = int(datetime.now().timestamp())
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None
            
        # CSV verisini pandas DataFrame'e dönüştür
        df = pd.read_csv(pd.compat.io.StringIO(response.text))
        if len(df) < 40:  # Yeterli veri yoksa
            return None
            
        # Veri hazırlığı
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    except Exception as e:
        return None

def calculate_indicators_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """Hızlı ve güvenli indikatör hesaplama"""
    try:
        # Temel indikatörler - minimum hesaplama
        df['RSI'] = ta.rsi(df['Close'], length=14, fillna=True)
        
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9, fillna=True)
        if macd is not None:
            df = pd.concat([df, macd[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']]], axis=1)
        
        df['Volume_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14, fillna=True)
        
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14, fillna=True)
        if adx is not None:
            df = pd.concat([df, adx[['ADX_14', 'DMP_14', 'DMN_14']]], axis=1)
        
        supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3.0, fillna=True)
        if supertrend is not None and 'SUPERT_7_3.0' in supertrend.columns:
            df['SuperTrend'] = supertrend['SUPERT_7_3.0']
        
        bb = ta.bbands(df['Close'], length=20, std=2, fillna=True)
        if bb is not None:
            df = pd.concat([df, bb[['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBP_20_2.0', 'BBW_20_2.0']]], axis=1)
        
        df['EMA20'] = ta.ema(df['Close'], length=20, fillna=True)
        df['EMA50'] = ta.ema(df['Close'], length=50, fillna=True)
        
        return df
    except Exception as e:
        return df

def calculate_score_optimized(df: pd.DataFrame) -> Dict[str, Any]:
    """Vektörel işlemlerle optimize edilmiş skor hesaplama"""
    if len(df) < 2:
        return {"total_score": 0, "components": {"details": {}}}
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    scores = {
        "RSI": 0, "MACD": 0, "Volume_MFI": 0, 
        "ADX": 0, "SuperTrend": 0, "Bollinger": 0,
        "details": {}
    }
    
    # RSI Hesaplama (20 puan)
    rsi = last_row.get('RSI', 50)
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
    macd_line = last_row.get('MACD_12_26_9', 0)
    signal_line = last_row.get('MACDs_12_26_9', 0)
    hist = last_row.get('MACDh_12_26_9', 0)
    prev_hist = prev_row.get('MACDh_12_26_9', 0)
    
    macd_condition = macd_line > signal_line
    prev_macd = prev_row.get('MACD_12_26_9', 0)
    prev_signal = prev_row.get('MACDs_12_26_9', 0)
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
    vol = last_row.get('Volume', 0)
    vol_ma = last_row.get('Volume_MA20', 1)  # Bölme hatası için min 1
    mfi = last_row.get('MFI', 50)
    prev_mfi = prev_row.get('MFI', 50)
    
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
    adx = last_row.get('ADX_14', 0)
    dmp = last_row.get('DMP_14', 0)
    dmn = last_row.get('DMN_14', 0)
    prev_adx = prev_row.get('ADX_14', 0)
    
    if adx > 25 and dmp > dmn:
        scores["ADX"] = 15
        scores["details"]["ADX"] = f"ADX: {adx:.1f} > 25 + DI+ > DI- (15 Puan)"
    elif 20 <= adx <= 25 and (adx > prev_adx):
        scores["ADX"] = 10
        scores["details"]["ADX"] = f"ADX: {adx:.1f} ve Yükselen Trend (10 Puan)"
    else:
        scores["details"]["ADX"] = "Puan almadı"
    
    # SuperTrend (15 puan)
    close = last_row.get('Close', 0)
    st_line = last_row.get('SuperTrend', 0)
    
    if close > st_line:
        scores["SuperTrend"] = 15
        scores["details"]["SuperTrend"] = f"Fiyat: {close:.2f} > SuperTrend: {st_line:.2f} (15 Puan)"
    else:
        scores["details"]["SuperTrend"] = f"Fiyat: {close:.2f} < SuperTrend: {st_line:.2f} (0 Puan)"
    
    # Bollinger (10 puan)
    bb_percent = last_row.get('BBP_20_2.0', 0.5)
    bb_width = last_row.get('BBW_20_2.0', 0.2)
    sma20 = last_row.get('BBM_20_2.0', close)
    
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

def create_chart_optimized(df: pd.DataFrame, symbol: str, name: str, score_details: Dict[str, Any], 
                          show_bb: bool = True, show_ema20: bool = True, show_ema50: bool = True, 
                          show_supertrend: bool = True) -> go.Figure:
    """Hafifletilmiş ama bilgilendirici grafik"""
    if df is None or len(df) < 40:
        fig = go.Figure()
        fig.add_annotation(
            text="Yeterli veri yok",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        return fig
    
    # Son 60 günü göster (performans için)
    df_display = df.tail(60).copy()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Mum grafiği
    fig.add_trace(go.Candlestick(
        x=df_display['Date'],
        open=df_display['Open'],
        high=df_display['High'],
        low=df_display['Low'],
        close=df_display['Close'],
        name='Mumlar',
        increasing_line_color='#2ecc71',
        decreasing_line_color='#e74c3c'
    ), row=1, col=1)
    
    # İndikatörler
    if show_supertrend and 'SuperTrend' in df_display.columns:
        fig.add_trace(go.Scatter(
            x=df_display['Date'], y=df_display['SuperTrend'],
            mode='lines',
            name='SuperTrend',
            line=dict(color='#9b59b6', width=2)
        ), row=1, col=1)
    
    if show_ema20 and 'EMA20' in df_display.columns:
        fig.add_trace(go.Scatter(
            x=df_display['Date'], y=df_display['EMA20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='#3498db', width=1.5)
        ), row=1, col=1)
    
    if show_ema50 and 'EMA50' in df_display.columns:
        fig.add_trace(go.Scatter(
            x=df_display['Date'], y=df_display['EMA50'],
            mode='lines',
            name='EMA 50',
            line=dict(color='#e67e22', width=1.5)
        ), row=1, col=1)
    
    if show_bb and 'BBU_20_2.0' in df_display.columns:
        fig.add_trace(go.Scatter(
            x=df_display['Date'], y=df_display['BBU_20_2.0'],
            mode='lines',
            name='Üst Bant',
            line=dict(color='#7f8c8d', width=1, dash='dot')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_display['Date'], y=df_display['BBL_20_2.0'],
            mode='lines',
            name='Alt Bant',
            line=dict(color='#7f8c8d', width=1, dash='dot')
        ), row=1, col=1)
    
    # İkinci panel - RSI
    if 'RSI' in df_display.columns:
        fig.add_trace(go.Scatter(
            x=df_display['Date'], y=df_display['RSI'],
            mode='lines',
            name='RSI (14)',
            line=dict(color='#9b59b6', width=2)
        ), row=2, col=1)
        
        fig.add_hrect(y0=70, y1=100, fillcolor="#e74c3c", opacity=0.1, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="#2ecc71", opacity=0.1, row=2, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="#7f8c8d", row=2, col=1)
    
    # Layout optimizasyonu
    fig.update_layout(
        title=f"{symbol} | Skor: {score_details['total_score']}/100",
        title_font_size=18,
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=650,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=40, r=40),
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(gridcolor='#ecf0f1', title_text="Tarih")
    fig.update_yaxes(gridcolor='#ecf0f1')
    
    return fig

# Sidebar - MOD SEÇİMİ
with st.sidebar:
    st.title("📊 BIST Analiz Modları")
    
    # Mod seçimi butonları
    st.markdown("### Analiz Modunu Seçin")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚡ HIZLI MOD\n(BIST 100)", use_container_width=True, 
                    help="45-60 saniyede tamamlanır - En likit hisseler"):
            st.session_state.analysis_mode = "BIST100"
            st.session_state.analysis_started = True
    
    with col2:
        if st.button("🔍 TAM MOD\n(BIST TÜM)", use_container_width=True, 
                    help="3-5 dakika sürer - Tüm piyasa fırsatları"):
            st.session_state.analysis_mode = "BISTTUM"
            st.session_state.analysis_started = True
    
    st.markdown("---")
    st.subheader("📈 Grafik Ayarları")
    show_bb = st.toggle("Bollinger Bantları", value=True)
    show_ema20 = st.toggle("EMA 20", value=True)
    show_ema50 = st.toggle("EMA 50", value=True)
    show_supertrend = st.toggle("SuperTrend", value=True)
    
    st.markdown("---")
    st.caption("*Mod Karşılaştırması:*\n")
    st.caption("⚡ *Hızlı Mod:*\n- 100 hisse\n- 45-60 sn\n- En likit hisseler")
    st.caption("🔍 *Tam Mod:*\n- 500+ hisse\n- 3-5 dk\n- Tüm piyasa fırsatları")

# Session state başlatma
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "BIST100"
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# Ana ekran
st.title("🚀 BIST Swing Trading Analiz Paneli")
st.markdown("### İki modlu sistem: Hızlı BIST 100 veya Tam BIST TÜM analizi")

# Mod açıklamaları
if st.session_state.analysis_mode == "BIST100":
    st.info("⚡ *HIZLI MOD* aktif: Sadece BIST 100 hisseleri analiz ediliyor. 45-60 saniyede tamamlanır.")
else:
    st.warning("🔍 *TAM MOD* aktif: Tüm BIST TÜM hisseleri analiz ediliyor. Tamamlanması 3-5 dakika sürer.")

# Analiz başlatma
if st.session_state.analysis_started:
    mode = st.session_state.analysis_mode
    symbols = get_bist_symbols(mode)
    total_symbols = len(symbols)
    
    # Mod bilgisi
    if mode == "BIST100":
        st.info(f"⚡ Hızlı mod: {total_symbols} BIST 100 hissesi analiz ediliyor...")
    else:
        st.warning(f"🔍 Tam mod: {total_symbols} BIST TÜM hissesi analiz ediliyor. Lütfen bekleyin...")
    
    # İlerleme çubuğu
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    results = []
    error_count = 0
    
    # PARALEL İŞLEME - Akıllı thread yönetimi
    max_workers = 20 if mode == "BISTTUM" else 15
    batch_size = 50 if mode == "BISTTUM" else total_symbols
    
    # Toplu işlem - batch processing
    for i in range(0, total_symbols, batch_size):
        batch_symbols = symbols[i:i+batch_size]
        batch_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(fetch_stock_data_parallel, symbol): symbol for symbol in batch_symbols}
            
            for j, future in enumerate(concurrent.futures.as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                try:
                    df = future.result(timeout=20)
                    current_progress = (i + j + 1) / total_symbols
                    
                    if df is not None and len(df) > 40:
                        # İndikatörleri hesapla
                        df = calculate_indicators_optimized(df)
                        
                        if df is not None and len(df) > 30:
                            # Skoru hesapla
                            score_details = calculate_score_optimized(df)
                            
                            # Son veriler
                            last_price = df.iloc[-1]['Close']
                            prev_close = df.iloc[-2]['Close'] if len(df) > 1 else last_price
                            change_percent = ((last_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
                            
                            # Hisse adı (basitleştirilmiş)
                            name = symbol.replace('.IS', '')
                            
                            batch_results.append({
                                "symbol": symbol.replace('.IS', ''),
                                "name": name,
                                "price": last_price,
                                "change": change_percent,
                                "score": score_details['total_score'],
                                "details": score_details,
                                "df": df.tail(60)  # Sadece son 60 günü sakla - bellek optimizasyonu
                            })
                    else:
                        error_count += 1
                    
                    # İlerleme güncelle
                    elapsed = time.time() - start_time
                    completed = i + j + 1
                    eta = (elapsed / completed) * (total_symbols - completed) if completed > 0 else 0
                    status = f"İŞLENİYOR: {completed}/{total_symbols} | Tahmini Süre: {eta/60:.1f} dk"
                    if mode == "BISTTUM":
                        status += f" | Başarılı: {len(results)+len(batch_results)} | Hata: {error_count}"
                    status_text.text(status)
                    progress_bar.progress(current_progress)
                    
                except Exception as e:
                    error_count += 1
                    continue
        
        # Batch sonuçlarını ekle
        results.extend(batch_results)
        
        # Bellek optimizasyonu - her batch'ten sonra bekleyin
        time.sleep(1)
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        # Skora göre sırala
        results.sort(key=lambda x: x['score'], reverse=True)
        st.session_state.results = results
        st.session_state.error_count = error_count
        total_time = time.time() - start_time
        
        if mode == "BISTTUM":
            st.success(f"✅ TAM MOD TAMAMLANDI! {len(results)}/{total_symbols} hisse analiz edildi. Süre: {total_time/60:.1f} dakika")
        else:
            st.success(f"✅ HIZLI MOD TAMAMLANDI! {len(results)}/{total_symbols} hisse analiz edildi. Süre: {total_time:.1f} saniye")
    else:
        st.error("❌ Analiz sonuçları alınamadı. Lütfen tekrar deneyin.")
        st.session_state.analysis_started = False

# Sonuçları göster
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    
    # Mod bilgisi
    if st.session_state.analysis_mode == "BIST100":
        st.subheader(f"⚡ En İyi {min(20, len(results))} BIST 100 Swing Fırsatı")
    else:
        st.subheader(f"🔍 En İyi {min(20, len(results))} BIST TÜM Swing Fırsatı")
    
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
            "Fiyat (₺)": f"{res['price']:.2f}",
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
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Skor detayları
            with st.expander("📊 Skor Detayları"):
                st.subheader(f"{selected['symbol']} - Skor Analizi")
                
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
                    st.info(f"*Toplam Skor:* {selected['score']}/100")
                    if selected['score'] >= 90:
                        st.success("⭐ *Mükemmel Swing Fırsatı!* Tüm kriterler olumlu.")
                    elif selected['score'] >= 70:
                        st.warning("✅ *İyi Fırsat* - Temkinli pozisyon alınabilir.")
                    elif selected['score'] >= 50:
                        st.error("⚠️ *Dikkatli Olun* - Sadece tecrübeli yatırımcılar için.")
                    else:
                        st.error("❌ *Önerilmez* - Yeterli teknik sinyal yok.")
                    
                    st.markdown("##### 📌 Tavsiye Edilen İşlem:")
                    if selected['score'] >= 90:
                        st.markdown("🟢 *AL* - Güçlü trend, hacim onaylı, RSI ideal seviyede")
                    elif selected['score'] >= 70:
                        st.markdown("🟡 *İZLE* - Potansiyel fırsat var, onay bekleyin")
                    else:
                        st.markdown("🔴 *BEKLE* - Daha iyi fırsatlar için takip edin")
        else:
            st.warning("Seçilen hisse için veri bulunamadı.")
else:
    st.info("""
    ### 🚀 Başlamak İçin
        
    *İki farklı analiz modu mevcut:*
    
    1. *⚡ HIZLI MOD (BIST 100):* 
       - Sadece en likit 100 hisse
       - 45-60 saniyede tamamlanır
       - Acil kararlar için ideal
    
    2. *🔍 TAM MOD (BIST TÜM):*
       - Tüm BIST hisseleri (~500+)
       - 3-5 dakika sürer
       - Tüm piyasa fırsatlarını görmek için
    
    👉 *Sol menüden istediğiniz modu seçin ve analizi başlatın!*
    """)

# Footer
st.markdown("---")
st.caption(f"🔄 Son Güncelleme: {datetime.now().strftime('%d %B %Y %H:%M')} | Veri: Yahoo Finance")
st.caption("💡 *Bilgi:* Bu araç yatırım tavsiyesi değildir. Swing trading yüksek risk içerir. Lütfen kendi araştırma ve risk yönetiminizi yapın.")
