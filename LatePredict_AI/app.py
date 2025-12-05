import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1. 내부용 학습 데이터 & 모델
# -----------------------------

def build_training_data():
    data = [
        [1.2, 23, 0, 0],   # 23시 (밤)
        [3.5, 2, 1, 1],    # 02시
        [2.1, 1, 0, 0],
        [4.0, 3, 1, 1],
        [1.8, 0, 0, 0],    # 00시
        [5.2, 22, 1, 1],   # 22시
        [2.5, 6, 0, 0],
        [3.0, 5, 1, 1],
        [4.2, 2, 1, 1],
        [1.0, 7, 0, 0]
    ]
    df = pd.DataFrame(data, columns=['distance', 'sleep_time_24', 'weather', 'late'])
    return df

@st.cache_resource
def train_model():
    df = build_training_data()

    X = df[['distance', 'sleep_time_24', 'weather']]
    y = df['late']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


# -----------------------------
# 2. Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="학생 지각 확률 예측기", page_icon="⏰", layout="centered")

    st.title("⏰ 학생 지각 확률 예측기")

    st.write("아래 질문 3개에 답하면, 오늘 지각할 확률을 간단히 예측할 수 있어요.")

    model = train_model()

    st.markdown("---")

    # ===== 입력 =====
    st.subheader("Q1. 집에서 학교까지 통학 거리는 몇 km인가요?")
    distance = st.number_input(
        "통학 거리 (km)",
        min_value=0.0,
        max_value=20.0,
        value=3.5,
        step=0.1,
    )

    st.markdown("---")

    st.subheader("Q2. 어제 몇 시에 잠들었나요?")
    col1, col2 = st.columns(2)

    with col1:
        sleep_hour = st.number_input(
            "시간 (1~12)",
            min_value=0,
            max_value=12,
            value=11,
            step=1,
        )
    with col2:
        am_pm = st.radio("오전/오후 선택", ["오전(AM)", "오후(PM)"])

    # 12시간제 → 24시간 변환
    # ------------------------
    if am_pm == "오전(AM)":
        sleep_24 = sleep_hour % 24  # 12 → 0, 1→1
    else:
        sleep_24 = (sleep_hour % 12) + 12  # 12→12, 1→13, 11→23

    st.caption(f" → 24시간 기준 환산: **{sleep_24}시**")

    st.markdown("---")

    st.subheader("Q3. 날씨는 어떤가요?")
    weather_label = st.selectbox(
        "날씨 선택",
        ["맑음", "비/눈"],
    )
    weather = 0 if weather_label == "맑음" else 1

    st.markdown("---")

    # ===== 결과 =====
    if st.button("📊 지각 확률 예측하기"):
        new_data = pd.DataFrame(
            [[distance, sleep_24, weather]],
            columns=['distance', 'sleep_time_24', 'weather']
        )
        prob = model.predict_proba(new_data)[0][1]
        percent = prob * 100

        st.subheader("예측 결과")
        st.markdown(f"### 👉 지각 확률: **{percent:.1f}%**")

    else:
        st.info("지각 확률을 확인하려면 버튼을 눌러주세요.")


if __name__ == "__main__":
    main()
