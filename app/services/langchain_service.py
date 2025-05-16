from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from app.core.config import settings  # Assuming settings is loaded from your .env
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from typing import Dict

class LangchainService: 
    def __init__(self):
        llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN 
        )

        self.chat_model = ChatHuggingFace(llm=llm)

    def summarize(self,context: Dict):
        messages = [
            SystemMessage(content="You're a smart summarizer that understand context of a STEM research paper"),
            HumanMessage(
                content=f"I have a paper with title {context['title']} and abstract {context['abstract']}"
            ),
            SystemMessage(content="I want you to summarize the content of the research paper so I can understand it easily.")
        ]

        ai_msg = self.chat_model.invoke(messages)

        return ai_msg