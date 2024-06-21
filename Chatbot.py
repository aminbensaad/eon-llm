import streamlit as st
from transformers import BertTokenizer, BertForMaskedLM
import torch


def answer_prompt(prompt: str) -> str:
    model_name = "bert-base-uncased"
    #model_name = "./bert-finetuned-squad/checkpoint-33276"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(inputs, num_return_sequences=1)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # answer_start_scores, answer_end_scores = model(**inputs).values()

    # answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    # answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    # answer = tokenizer.convert_tokens_to_string(
    #     tokenizer.convert_ids_to_tokens(
    #         inputs['input_ids'][answer_start:answer_end]
    #         )
    # )

    return answer


# with st.sidebar:
#     openai_api_key = st.text_input(
#         "OpenAI API Key", key="chatbot_api_key", type="password"
#     )
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

col1, _, col2 = st.columns([2, 1, 15])
with col1:
    st.markdown(
        """
        <div style='height:30px;'></div>
    """,
        unsafe_allow_html=True,
    )
    st.image("assets/eon-logo.png", width=120, output_format="PNG")
with col2:
    st.title("Chatbot")

st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    # client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo", messages=st.session_state.messages
    # )
    # msg = response.choices[0].message.content
    msg = answer_prompt(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
