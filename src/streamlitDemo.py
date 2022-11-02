import generaterMethod as gen
import modelLoader as load
import streamlit as st

@st.cache
def get_model(batch_size: int):
    model, tokenizer = load.gdLoadModelForSequenceClassification('beomi/KcELECTRA-base-v2022')
    gen_model, gen_tokenizer = load.gdLoadModelForCausalLM('skt/kogpt2-base-v2')
    return (model, tokenizer, gen_model, gen_tokenizer, batch_size)

generater = gen.generater(**get_model(16))

st.title("문장 순화기")

input = st.text_area("입력", placeholder="순화할 비윤리적 문장")

if st.button("변경", help="해당 입력을 순화합니다."):
    st.text_area("출력", value=generater.makeMoralText(input), disabled=True)