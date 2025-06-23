import streamlit as st
import json
import os
import pandas as pd

from dotenv import load_dotenv
from langfuse import observe, Langfuse
from langfuse.openai import openai
from pycaret.regression import load_model, predict_model

MODEL_NAME = "best_model"

load_dotenv()



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

########

# Tytuł aplikacji
#st.title("🏃 RunAlyze AI 🤖")
#st.subheader("Oszacuję dla Ciebie czas w jakim mógłbyś przebiec półmaraton (~21km) jeśli się postarasz")

st.markdown('<h1 class="centered-title">🏃 RunAlyze AI 📊</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="centered-title">Oszacuję dla Ciebie czas w jakim mógłbyś przebiec półmaraton (~21km) jeśli się postarasz! ⚡</h3>', unsafe_allow_html=True)


if "text_area" not in st.session_state:
    st.session_state["text_area"] = ""
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
if "text_area" not in st.session_state:
    st.session_state["text_area"] = ""
# Ustaw placeholder (pokaże się, gdy pole jest puste)
placeholder_text = "Pochwal się..."

# Pole tekstowe
text = st.text_area(
    "Witaj biegaczu. Przedstaw się, ile masz lat, podaj swoją płeć oraz tempo biegu na 5 km.",
    value=st.session_state.get("text_area", ""),
    key="text_area",
    placeholder=placeholder_text
)

if st.button("Szacowanko 🎯"):
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

            #plec = extracted.get("Płeć")
            #wiek = extracted.get("Wiek")
            #tempo = extracted.get("5 km Tempo")
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
            #if not isinstance(wiek, int) or not (10 <= wiek <= 100):
                #messages.append("⚠️ Wiek poza zakresem.")
                #valid = False

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
            st.session_state["submitted"] = True
            st.success(f" π razy 👁️ wyjdzie {formatted_time}")

def reset():
    st.session_state["text_area"] = ""
    st.session_state["submitted"] = False

if st.session_state["submitted"]:
    st.button("🔄 Odśwież", on_click=reset)