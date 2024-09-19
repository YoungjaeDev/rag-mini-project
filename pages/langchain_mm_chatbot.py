# 필요한 라이브러리 임포트
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image
import streamlit as st
from streamlit_chat import message
from langchain import LLMChain, PromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List, Any

MODEL_OPTIONS = {
    # 모델 정보 유지
    "LLaVA-Next 8B": {
        "processor": "llava-hf/llama3-llava-next-8b-hf",
        "model": "llava-hf/llama3-llava-next-8b-hf",
        "processor_class": LlavaNextProcessor,
        "model_class": LlavaNextForConditionalGeneration,
        "torch_dtype": torch.float16
    },
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
    model_info = MODEL_OPTIONS[model_name]
    processor = model_info["processor_class"].from_pretrained(model_info["processor"])
    model = model_info["model_class"].from_pretrained(
        model_info["model"],
        torch_dtype=torch.float16,
        device_map='auto'
    )
    return processor, model

# LangChain용 커스텀 LLM 클래스
class CustomLLM(LLM):
    max_tokens: int = 512
    processor: Any = None
    model: Any = None
    image: Any = None

    def __init__(self, processor, model, image, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model
        self.image = image

    @property
    def _llm_type(self) -> str:
        return "CustomLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image'}
                ]
            }
        ]
        chat_prompt = self.processor.apply_chat_template(conversation, add_generic_prompt=True)
        inputs = self.processor(images=self.image, text=chat_prompt, return_tensors='pt').to(self.model.device, self.model.dtype)
        output = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        response = self.processor.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        return response

def generate_response_langchain(processor, model, image, user_input):
    llm = CustomLLM(processor=processor, model=model, image=image)

    prompt_template = PromptTemplate(
        input_variables=["input"],
        template="사용자: {input}\n어시스턴트:"
    )

    # LLMChain 생성
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    response = chain.run(input=user_input)

    return response

def show():
    st.title("RAG 1) 이미지-텍스트 멀티모달 챗봇 with LangChain")

    selected_model = st.sidebar.selectbox(
        "모델 선택",
        options=list(MODEL_OPTIONS.keys()),
        index=1
    )

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
                response = generate_response_langchain(processor, model, image, prompt)

            st.session_state.messages.append({"role": "assistant", "content": response})
            message(response)

if __name__ == "__main__":
    show()
