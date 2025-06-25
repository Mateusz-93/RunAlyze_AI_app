import streamlit as st
import json
import os
import pandas as pd

from dotenv import load_dotenv
from langfuse import observe, Langfuse
from langfuse.openai import openai
from pycaret.regression import load_model, predict_model

import boto3
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta


MODEL_NAME = "best_model"

load_dotenv()

s3 = boto3.client("s3")
BUCKET_NAME = "maratonymateusz"

# Tworzysz obiekt Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in os.environ:
        st.session_state["openai_api_key"] = os.environ["OPENAI_API_KEY"]
    else:
        st.info("Podaj swój klucz OpenAI:")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

@observe() # śledzenie przez Langfuse
def get_data_from_user_text_observed(message, model="gpt-4o"):
    prompt = """
    Jesteś pomocnikiem, któremu zostaną podane dane dotyczące płci, wieku oraz tempie biegu na 5 km. 
    <płeć>: dla mężczyzny oznacz jako "M". Dla kobiety oznacz jako "K". Jeżeli nie zostanie podane wprost to może po imieniu albo sposobie pisania uda Ci się ustalić płeć. Jeśli nie to zostaw puste.
    <wiek>: liczba lat, lub przelicz rok urodzenia.
    <5 km Tempo>: w minutach/km, np. 6:20 lub 6.20, jeśli ktoś poda czas biegu na 5km to przelicz
    Zwróć wynik jako poprawny JSON:
    {"Płeć": "...", "Wiek": ..., "5 km Tempo": ...}
    """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"```{message}```",
        },
    ]

    chat_completion = openai.chat.completions.create(
        response_format={"type": "json_object"},
        messages=messages,
        model=model,
    )
    resp = chat_completion.choices[0].message.content
    try:
        output = json.loads(resp)
    except:
        output = {"error": resp}
    return output

def get_data_from_users_observed(messages):
    results = []
    for message in messages:
        parsed = get_data_from_user_text_observed(message)
        results.append({"message": message, **parsed})
    return pd.DataFrame(results)

###############################################

@observe()  # będziemy mieć trace odnosnie wyniku, czyli full pakiet, dane wejściowe oraz wynik
def make_prediction_logged(data):
    with langfuse.start_as_current_span(
        name="półmaraton_predict",
        input=data.to_dict()
    ) as span:
        wynik = predict_model(halfmarathon_model, data=data)
        prediction_time = wynik["prediction_label"].values[0]
        formatted_time = format_seconds_to_hms(prediction_time)

        span.update(
            output={
                "prediction_seconds": prediction_time,
                "prediction_time_formatted": formatted_time
            }
        )
        return wynik

##################################################

@st.cache_resource
def load_halfmarathon_model():
    return load_model(MODEL_NAME)

halfmarathon_model = load_halfmarathon_model()

def convert_time_to_minutes(time_str):
    if isinstance(time_str, str):
        if ":" in time_str:
            m, s = map(int, time_str.strip().split(":"))
            return m + s / 60
        elif "." in time_str:
            try:
                m, sec_decimal = map(int, time_str.strip().split("."))
                return m + (sec_decimal / 100)
            except:
                pass
    return float(time_str)

def format_seconds_to_hms(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"

def sekundy_na_hms_tick(sek):
    return str(timedelta(seconds=int(sek)))



### Wczytanie danych z serwera
@st.cache_data
def load_halfmarathon_data():
    df_2023 = pd.read_csv(f"s3://{BUCKET_NAME}/data/halfmarathon_2023.csv")
    df_2024 = pd.read_csv(f"s3://{BUCKET_NAME}/data/halfmarathon_2024.csv")

    return df_2023, df_2024

#
# Rysowanie wykresu
#

def przygotuj_i_rysuj(
    df, 
    rok, 
    czas_uzytkownika_s=None, 
    wiek_uzytkownika=None, 
    plec_uzytkownika=None, 
    czas_hms_uzytkownika=None
):
    df = df[df["Rocznik"].between(1920, 2020)]
    df = df[df["Czas"].notna() & (df["Czas"] != 0)]

    df["Wiek"] = rok - df["Rocznik"]
    df["czas_s"] = df["Czas"].astype(float)
    df = df[df["czas_s"] > 0]
    df["czas_hms"] = df["czas_s"].apply(format_seconds_to_hms)

    kolor_map = {"M": "darkblue", "K": "deeppink"}

    fig = px.scatter(
        df,
        x="czas_s",
        y="Wiek",
        color="Płeć",
        color_discrete_map=kolor_map,
        labels={"czas_s": "Czas (s)", "Wiek": "Wiek"},
        opacity=0.2,
        height=600,
        custom_data=["Płeć", "czas_hms"]
    )

    fig.update_traces(hovertemplate=
                      "Płeć: %{customdata[0]}<br>Wiek: %{y}<br>Czas: %{customdata[1]}")

    tickvals = list(range(3600, 14401, 1800))  # co 30 min: od 1h do 4h
    ticktext = [sekundy_na_hms_tick(val) for val in tickvals]

    fig.update_layout(
        title=dict(
            text=f"Wyniki półmaratonu Wrocław - {rok}",
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[3600, 14400],
            title="Czas",
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=0.1,
            griddash='dash'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=0.1,
            title="Wiek",
            griddash='dash'
        )
    )
    fig.update_xaxes(layer="below traces") # osie pod wynikami
    fig.update_yaxes(layer="below traces")

    if all(v is not None for v in [czas_uzytkownika_s, wiek_uzytkownika, plec_uzytkownika, czas_hms_uzytkownika]):
        fig.add_trace(
            go.Scatter(
                x=[czas_uzytkownika_s],
                y=[wiek_uzytkownika],
                mode="markers+text",
                marker=dict(
                    symbol="star",
                    size=28,
                    color="gold",
                    line=dict(color="black", width=2)
                ),
                text=["Twój wynik!"],
                textposition="top center",
                textfont=dict(size=22, color="white", family="Arial"),
                name="Twój wynik",
                hovertemplate=(
                    f"Płeć: {plec_uzytkownika}<br>"
                    f"Wiek: {wiek_uzytkownika}<br>"
                    f"Czas: {czas_hms_uzytkownika}<extra></extra>"
                ),
                opacity=1.0,
                showlegend=True,
            )
        )
    st.plotly_chart(fig, use_container_width=True)

########

# Tytuł aplikacji
st.markdown('<h1 style="text-align:center">🏃 RunAlyze AI 📊</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align:center">Oszacuję dla Ciebie czas w jakim mógłbyś przebiec półmaraton (~21km) jeśli się postarasz! ⚡</h3>', unsafe_allow_html=True)


if "text_area" not in st.session_state:
    st.session_state["text_area"] = ""
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
# Ustaw placeholder (pokaże się, gdy pole jest puste)
placeholder_text = "Pochwal się..."

# Pole tekstowe
text = st.text_area(
    "Witaj biegaczu. Przedstaw się, ile masz lat, podaj swoją płeć oraz tempo biegu na 5 km.",
    value=st.session_state.get("text_area", ""),
    key="text_area",
    placeholder=placeholder_text
)

if st.button("Szacowanko 🤖"):
    if not text.strip():
        st.warning("Wprowadź dane przed kliknięciem!")
    else:
        
        with st.spinner("Komputer się grzeje by spełnić twe marzenie!"):
            extracted = get_data_from_users_observed([text])
            valid = True
            messages = []

            if "error" in extracted.columns:
                st.error(f"Błąd odczytu danych: {extracted.iloc[0]['error']}")
                st.stop()

            row = extracted.iloc[0]
            plec = row.get("Płeć")
            wiek = row.get("Wiek")
            tempo = row.get("5 km Tempo")

            try:
                tempo_float = convert_time_to_minutes(tempo)
                if not (3.0 <= tempo_float <= 12.0):
                    messages.append("⚠️ Tempo wygląda podejrzanie (zakres 3:00-12:00).")
                    valid = False
            except:
                messages.append("⚠️ Niepoprawny format tempa.")
                valid = False

            if plec not in ["M", "K"]:
                messages.append("⚠️ Nie udało się określić płci.")
                valid = False

            try:
                wiek = int(wiek)
                if not (10 <= wiek <= 100):
                    messages.append("⚠️ Wiek poza zakresem.")
                    valid = False
            except:
                messages.append("⚠️ Nie udało się odczytać wieku. Czy podałeś go jako liczbę?")
                valid = False

            if not valid:
                for msg in messages:
                    st.warning(msg)
                st.stop()

            st.subheader("Dane wyciągnięte z wiadomości:")
            col1, col2, col3 = st.columns(3)
            col1.metric("Płeć", plec)
            col2.metric("Wiek", wiek)
            col3.metric("Tempo 5 km", tempo)

            dane_biegacza = pd.DataFrame([{
                "Wiek": wiek,
                "Płeć": plec,
                "5 km Tempo": tempo_float
            }])

            #prediction = predict_model(halfmarathon_model, data=dane_biegacza)
            prediction = make_prediction_logged(dane_biegacza)
            prediction_time = prediction["prediction_label"].values[0]
            formatted_time = format_seconds_to_hms(prediction_time)
            st.session_state["formatted_time"] = formatted_time

            st.session_state["prediction_time"] = prediction_time
            st.session_state["wiek"] = wiek
            st.session_state["plec"] = plec
            
            st.session_state["submitted"] = True
if "prediction_time" in st.session_state and "formatted_time" in st.session_state:
    st.markdown(
        """
        <div style="text-align:center; margin-top:1em;">
            <p style="font-size:20px; color:green;">✅ Udało się wyciągnąć dane i oszacować czas!</p>
            <div style="font-size:28px; color:green; font-weight:bold;">🎯 Twój przewidywany czas: {}</div>
        </div>
        """.format(st.session_state["formatted_time"]),
        unsafe_allow_html=True
    )

def reset():
    st.session_state["text_area"] = ""
    st.session_state["submitted"] = False

if st.session_state["submitted"]:
    st.button("🔄 Odśwież", on_click=reset)



    st.subheader("📊 Porównaj swój wynik z innymi biegaczami:")
    rok = st.selectbox("Wybierz rok zawodów:", [2023, 2024])

    df_2023, df_2024 = load_halfmarathon_data()
    df = df_2023 if rok == 2023 else df_2024
    przygotuj_i_rysuj(
    df,
    rok,
    czas_uzytkownika_s=st.session_state["prediction_time"],
    wiek_uzytkownika=st.session_state["wiek"],
    plec_uzytkownika=st.session_state["plec"],
    czas_hms_uzytkownika=st.session_state["formatted_time"]
)