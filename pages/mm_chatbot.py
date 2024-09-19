# from transformers import pipeline
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image
import requests
import streamlit as st
from streamlit_chat import message

# sample prompt : 여자가 들고 있는 상품이 뭐야? 위 3개중에 선택해줘 1.도시락 2. 삼각깁밥 3. 음료수

MODEL_OPTIONS = {
    # https://huggingface.co/llava-hf/llama3-llava-next-8b-hf
    "LLaVA-Next 8B": {
        "processor": "llava-hf/llama3-llava-next-8b-hf",
        "model": "llava-hf/llama3-llava-next-8b-hf",
        "processor_class": LlavaNextProcessor,
        "model_class": LlavaNextForConditionalGeneration,
        "torch_dtype": torch.float16
    },
    # https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-chat-hf
    # pip install git+https://github.com/huggingface/transformers
    "LLaVA-OneVision-Chat Qwen2 7B": {
        "processor": "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
        "model": "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
        "processor_class": AutoProcessor,
        "model_class": LlavaOnevisionForConditionalGeneration,
        "torch_dtype": torch.float16
    }
}

@st.cache_resource
def init_model(model_name):
    # task: https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/&amp;num;transformers.pipeline.task
    model_info = MODEL_OPTIONS[model_name]
    processor = model_info["processor_class"].from_pretrained(model_info["processor"])
    model = model_info["model_class"].from_pretrained(
        model_info["model"], 
        torch_dtype=model_info["torch_dtype"],
        device_map='auto'
    )
    return processor, model

def generate_response(processor, model, image, text):
    conversation = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text},
                {'type': 'image'}
            ]
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generic_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device, model.dtype)
    output = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

def show():
    st.title("RAG 1) 이미지-텍스트 멀티모달 챗봇")
    
    selected_model = st.sidebar.selectbox(
        "모델 선택",
        options=list(MODEL_OPTIONS.keys()),
        index=1
    )
    
    # url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    # image = Image.open(requests.get(url, stream=True).raw)
    processor, model = init_model(selected_model)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_column_width=True)
        
        for i, chat in enumerate(st.session_state.messages):
            message(chat["content"], is_user=chat["role"] == "user", key=str(i))

        if prompt := st.chat_input("메시지를 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            message(prompt, is_user=True)

            with st.spinner("답변 생성 중..."):
                response = generate_response(processor, model, image, prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            message(response)
            
    
if __name__ == "__main__":
    show()
    