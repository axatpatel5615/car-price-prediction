import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="IPL Live Match Predictor", page_icon="🏏", layout="centered")

st.title("🏏 IPL Win Probability Engine")
st.markdown("Adjust live chasing metrics below to estimate win/loss paths instantly.")
st.write("---")

@st.cache_resource
def load_model():
    with open('ipl_probability_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

pipe = load_model()

teams = ['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bengaluru', 'Kolkata Knight Riders', 'Rajasthan Royals', 'Sunrisers Hyderabad', 'Delhi Capitals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans']

# Extracting prominent venues present in the dataset context arrays
venues = ['Wankhede Stadium', 'Eden Gardens', 'MA Chidambaram Stadium, Chepauk', 'M Chinnaswamy Stadium', 'Arun Jaitley Stadium', 'Rajiv Gandhi International Stadium', 'Narendra Modi Stadium', 'Punjab Cricket Association IS Bindra Stadium']

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team (Chasing)', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted([t for t in teams if t != batting_team]))

selected_venue = st.selectbox('Select Stadium Venue', sorted(venues))
target = st.number_input('Target Score Set', min_value=50, max_value=300, value=180)

st.write("---")
col3, col4, col5 = st.columns(3)
with col3:
    current_score = st.number_input('Current Score', min_value=0, max_value=300, value=100)
with col4:
    overs_completed = st.slider('Overs Completed', min_value=0.0, max_value=19.5, value=10.0, step=0.1, format="%g")
with col5:
    wickets_out = st.slider('Wickets Lost', min_value=0, max_value=9, value=3)

if st.button('Calculate Probabilities', type='primary', use_container_width=True):
    runs_left = target - current_score
    # Map raw floating point overs (e.g. 10.3) directly to absolute count weights
    balls_completed = int(overs_completed) * 6 + int((overs_completed % 1) * 10)
    balls_left = 120 - balls_completed
    wickets_left = 10 - wickets_out
    
    if balls_left > 0 and runs_left >= 0:
        crr = (current_score * 6) / balls_completed if balls_completed > 0 else 0
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame([{
            'batting_team': batting_team, 'bowling_team': bowling_team, 'venue': selected_venue,
            'runs_left': runs_left, 'balls_left': balls_left, 'wickets_left': wickets_left,
            'runs_target': target, 'crr': crr, 'rrr': rrr
        }])

        probabilities = pipe.predict_proba(input_df)[0]
        win_percentage = probabilities[1] * 100
        loss_percentage = probabilities[0] * 100

        st.markdown("### 📊 Prediction Metrics Output")
        st.write("---")
        w_col, l_col = st.columns(2)
        w_col.metric(label=f"🟢 {batting_team} Odds", value=f"{win_percentage:.1f}%")
        l_col.metric(label=f"🔴 {bowling_team} Odds", value=f"{loss_percentage:.1f}%")
        st.progress(int(win_percentage))
    else:
        st.error("Invalid state. Target exceeded or match matches parameters out of range boundaries.")