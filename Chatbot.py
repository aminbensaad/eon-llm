import importlib.util
import importlib.machinery
import json
import tempfile
import os
import types

import streamlit as st

from llm.scripts import model_ids

models = {}
model_type_labels = {
    "Base": "base",
    "SQuAD Tuned": "tuned",
    "GermanSQuAD Tuned": "Gtuned",
    "Local Model": "local",
}
for l, t in model_type_labels.items():
    models[l] = model_ids.model_IDs[t]

# currently selected model list
model_list = []


def create_initial_message():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]


if not "context" in st.session_state:
    st.session_state["context"] = "No context."


def fix_path(path: str) -> str:
    fixed_path = os.path.join("llm", "dummy", path)
    return os.path.normpath(fixed_path)


def generate_history_context():
    context = (
        "This is a chat between 'user' and 'assistant' with the following messages:\n"
    )
    message_history = st.session_state.messages
    for msg in message_history:
        role = msg["role"]
        content = msg["content"]
        context += f"{role}: {content}\n"
    return context


def load_model() -> types.ModuleType | None:
    current_model_type = model_type_labels[st.session_state["category_selection"]]
    current_model_id = st.session_state["model_selection"]
    script = model_ids.model_script_path(current_model_type, current_model_id)
    fixed_script_path = fix_path(script)

    if not os.path.exists(fixed_script_path):
        st.error(f"Unable to find model script at '{fixed_script_path}'")
        return

    module_name = model_ids.model_name_from_id(current_model_id).replace("-", "_")

    loader = importlib.machinery.SourceFileLoader(module_name, fixed_script_path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    if not spec:
        st.error(f"Unable to load model script at '{fixed_script_path}'")
        return None

    model_module = importlib.util.module_from_spec(spec)
    loader.exec_module(model_module)

    if not model_module:
        st.error(f"Failed to load '{current_model_id}'")
        return None
    st.info(f"Successfully loaded '{current_model_id}'")
    print(f"Successfully loaded '{current_model_id}': ", model_module)
    return model_module


def on_category_change():
    global model_list

    with st.sidebar:
        model_list = models.get(st.session_state["category_selection"], [])


def on_model_change():
    st.session_state["model_module"] = load_model()
    create_initial_message()


def on_context_input_changed():
    st.session_state["use_history"] = False


def on_history_context_changed():
    if not st.session_state["use_history"]:
        return

    st.session_state["context"] = generate_history_context()


def answer_question(question) -> str:
    model_module = st.session_state.get("model_module", None)
    if not model_module:
        return "Please select a model"

    model_input = {
        "data": [
            {
                "paragraphs": [
                    {
                        "context": st.session_state["context_input"],
                        "qas": [{"id": 0, "question": question}],
                    }
                ]
            }
        ]
    }
    print("Input: ", model_input)
    with tempfile.NamedTemporaryFile("w") as tmp_file:
        json.dump(model_input, tmp_file)
        tmp_file.flush()
        out_file = tempfile.NamedTemporaryFile()
        model_id = st.session_state["model_selection"]
        current_model_type = model_type_labels[st.session_state["category_selection"]]
        if current_model_type == "local":
            model_id = os.path.join(model_ids.local_model_dir, model_id)
            model_id = fix_path(model_id)
        model_module.main(model_id, tmp_file.name, out_file.name)
        return json.load(out_file).get("0", "Unable to generate answer")


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
    create_initial_message()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    answer = answer_question(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
    if st.session_state["use_history"]:
        st.session_state["context"] = generate_history_context()

with st.sidebar:
    current_category = st.selectbox(
        "Category",
        models.keys(),
        on_change=on_category_change,
        key="category_selection",
    )
    model_list = models.get(str(current_category), [])
    model_selection = st.selectbox(
        "Model",
        model_list,
        index=None,
        on_change=on_model_change,
        key="model_selection",
    )
    st.text_area(
        "Context",
        st.session_state["context"],
        key="context_input",
        on_change=on_context_input_changed,
        height=300,
    )
    st.checkbox(
        "Use history as context",
        key="use_history",
        on_change=on_history_context_changed,
    )
