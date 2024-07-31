import streamlit as st
from datetime import datetime
import requests
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, TextContentItem, ImageContentItem, ImageUrl, ImageDetailLevel
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from azure.core.credentials import AzureKeyCredential
import time
import ai_utils
import threading
from threading import current_thread
import base64
from io import StringIO



results = [None, None]
threadErrors=[]
base64_url = None
### Utility functions
def ai_inference_stream(ctx, 
                        prompt, 
                        index,
                        uploaded_img):
    add_script_run_ctx(threading.current_thread(), ctx)
    deployment=models[index]["deployment"]
    endpoint = models[index]["endpoint"]+"/openai/deployments/"+deployment
    api_key = models[index]["key"]
    if models[index]["model_type"] != "AOAI":
        client = ChatCompletionsClient(
            endpoint=models[index]["endpoint"],
            credential=AzureKeyCredential(api_key),
        )
    
    else:
        client = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(""),  # Pass in an empty value.
                headers={"api-key": api_key},
                api_version="2024-02-15-preview",  # AOAI api-version. Update as needed.
            )
        
    start_time = datetime.now()
    try:
        usercontent = [TextContentItem(text=prompt)]
        if uploaded_img is not None:
            usercontent.append(ImageContentItem(
                            image_url=ImageUrl(url=uploaded_img))
                            )
        stream = client.complete(
            messages=[
                SystemMessage(content=st.session_state["system_prompt"]),
                UserMessage(content=usercontent),
            ],
            stream=True,
            model_extras={} if st.session_state["num_tokens"]=="" else {"max_tokens":int(st.session_state["num_tokens"])},
        )

        assistant_reply = ""
        assistant_reply_box=containers[index].empty()

        first_token = None
        
        for event in stream:
            timers[index].empty()
            timers[index].markdown(f"#### Timer: {str(datetime.now()-start_time)[:-3]}")
            delta = event.choices[0].delta if event.choices and event.choices[0].delta is not None else None
            if delta and delta.content:
                if first_token is None:
                    first_token = datetime.now()
                assistant_reply_box.empty()
                assistant_reply += delta.content
                assistant_reply_box.markdown(assistant_reply)
        
        client.close()

        end_time = datetime.now()
        
        results[index]= {"response": assistant_reply,
                "total_time":end_time-start_time,
                "time_first_token":first_token-start_time,
                "output_tokens":ai_utils.num_tokens_from_string(assistant_reply, "gpt-4o")}
    except Exception as e:
        threadErrors.append([repr(e), current_thread().name])
        print(e)


def reset_conversation():
    container1.empty()
    container2.empty()
    prompt_container.empty()
    col1.empty()
    col2.empty()



st.set_page_config(page_title="Model Comparison", 
                   layout="wide",
                   initial_sidebar_state="expanded")
st.title('Azure AI - LLM Comparison')

headcol1, headcol2=st.columns([0.7,3])
clear = headcol1.button('New Comparison 🔄', on_click=reset_conversation)
prompt_container = headcol2.container()

col1, col2 = st.columns(2)

for i in range(2):
    if f"model_{i+1}" not in st.session_state:
        st.session_state[f"model_{i+1}"] = f"Model {i+1}"

with col1:
    title1=st.subheader(st.session_state["model_1"])
    container1 = st.container(height=300)
    timer1 = st.markdown("#### ⏱️ Timer: 0:00:00")

    
with col2:
    title2 = st.subheader(st.session_state["model_2"])
    container2 = st.container(height=300)
    timer2 = st.markdown("#### ⏱️ Timer: 0:00:00")
        

setup = st.sidebar
setup.subheader("Settings")
with setup.form("Model settings"):
    for i in range(2):
        st.subheader(f"Model {i+1}")
        collist = st.columns([2.5,2])
        with collist[0]:
            st.selectbox(f"Model", 
                         ["AOAI", "Azure MaaS", "Azure MaaP"], 
                         key=f"model_type_{i+1}",
                         label_visibility="collapsed",
                         placeholder="Select a model")
        
        with collist[1]:
            with st.popover(f"Settings"):
                st.text_input(f"Display Name", key=f"model_name_{i+1}")
                st.text_input("Endpoint URL", key=f"url_{i+1}",value="https://vh-oai-east2.openai.azure.com")
                st.text_input("API Key", type="password", key=f"key_{i+1}", value="57ad3c90d0f441fdbd7dffda4ae6601d")
                st.text_input("Deployment (AOAI Only)", 
                              key=f"deployment_{i+1}", 
                              help="Deployment name for AOAI models",
                              value="gpt-4o-global")

    
    st.text_area("System Prompt", value="You are a helpful AI Assistant", key="system_prompt")
    st.text_input("Max tokens", value=200, key="num_tokens")
    submit = st.form_submit_button("Save", type="primary")

help = """Supported models are those available on Azure, this includes:
        \n - Azure OpenAI models
        \n - Azure AI serverless inference models (LLama2, Cohere, Phi-3 etc.)
        \n - Azure managed endpoints (**ONLY** Llama3 instruct, Phi-3 family, Mixtral family)"""

setup.subheader("(Optional) Image Input")
uploaded_file = setup.file_uploader("Use Image input", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Convert bytes to base64
    base64_data = base64.b64encode(bytes_data).decode('utf-8')
    base64_url = f"data:image/png;base64,{base64_data}" 
    st.session_state["image"] = base64_url
    setup.image(uploaded_file, use_column_width=True)
else:
    st.session_state["image"] = None
setup.markdown(help)

if submit:
    for i in range(2):
        if st.session_state[f"model_name_{i+1}"]!="":
            st.session_state[f"model_{i+1}"] = st.session_state[f"model_name_{i+1}"]
        elif st.session_state[f"model_type_{i+1}"]!="":
            st.session_state[f"model_{i+1}"] = st.session_state[f"model_type_{i+1}"]
    st.rerun()
    

css = '''
<style>
    [data-testid="stSidebar"]{
        min-width: 100px;
        max-width: 25%;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True) 

timers = [timer1, timer2]
containers = [container1, container2]
columns = [col1, col2]
missing = False
bytes_data = None

if prompt := st.chat_input("Enter a prompt"):
    prompt_container.empty()
    
    if st.session_state.get('image') is not None:
        pccol1, pccol2 = prompt_container.columns([1,3])
        pccol2.image(uploaded_file, width=150)
        pccol1.markdown(f"**Prompt:** {prompt}")
    else:
        prompt_container.markdown(f" **Prompt:** {prompt}")
    models = [{"endpoint":st.session_state[f"url_{i+1}"],
               "key":st.session_state[f"key_{i+1}"],
               "deployment":st.session_state[f"deployment_{i+1}"],
               "model":st.session_state[f"model_{i+1}"],
               "model_type":st.session_state[f"model_type_{i+1}"]} 
               for i in range(2)]
    
    for i in range(2):
        if models[i]["endpoint"]=="" or models[i]["key"]=="":
            missing = True
    
    
    if missing:
        prompt_container.error("Please fill in endpoint and API Key for both models in the settings tab")
    else:
        ctx = get_script_run_ctx()
        threads = []
        for i in range(2):
            thread = threading.Thread(target=ai_inference_stream, args=(ctx,prompt,i, base64_url))
            thread.start()
            threads.append(thread)
        
        for t in threads:
            t.join()
        
        if len(threadErrors)>0:
            for error in threadErrors:
                headcol2.error(f"**Error:** {error[0]} in thread {error[1]}")
        
        else:
            for i in range(2):
                if results[i] is not None:
                    with columns[i]:
                        st.write(f"Time to first token: {results[i]['time_first_token'].total_seconds()} seconds")
                        st.write(f"Output Tokens: {results[i]['output_tokens']}")
                        st.write(f"Tokens per second: {float(results[i]['output_tokens'])/results[i]['total_time'].total_seconds()}")
        
    