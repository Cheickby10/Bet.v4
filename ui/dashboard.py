import streamlit as st
import matplotlib.pyplot as plt
from analytics.heatmap import mines_safety_map
from rl.trainer import train_deep

def show():
    st.title("BetMind AI — Advanced Dashboard")

    tab1, tab2 = st.tabs(["Safest Path", "Deep RL"])

    with tab1:
        if st.button("Generate Safety Heatmap"):
            heat = mines_safety_map()
            fig, ax = plt.subplots()
            ax.imshow(heat)
            st.pyplot(fig)

    with tab2:
        if st.button("Train Deep RL Agent"):
            score = train_deep()
            st.success(f"Training completed — Score {score}")
