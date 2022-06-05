import streamlit as st

from predict import load_model, get_prediction

from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# Streamlit App 만들기
def main():
    st.title("Open Domain Question Answering")

    model, tokenizer = load_model()
    model.eval()

    # query 입력
    text_input = st.text_input("질문을 입력해주세요")
    
    if text_input: 
        # 예측
        answer = get_prediction(model, tokenizer, text_input)
        # 예측 결과 출력
        answer_text = answer[0]['prediction_text']
        st.info(f'검색 결과 : **{answer_text}**')

main()
