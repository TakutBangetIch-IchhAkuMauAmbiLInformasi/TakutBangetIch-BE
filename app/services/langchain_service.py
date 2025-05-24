from typing import List
from transformers import pipeline
from app.models.search import SearchResult
from app.core.config import settings


class LangchainService:
    def __init__(self):
        self.summarizer = pipeline(
            "text-generation", 
            model="Qwen/Qwen2.5-0.5B", 
            token=settings.HUGGINGFACEHUB_API_TOKEN
        )

    def summarize(
        self, query: str, context: List[SearchResult], top_k: int = 3
    ) -> str:
        
        input_texts = []
        dois = []
        titles = ""
        for i, paper in enumerate(context[:top_k]):
            paper_info = (
                f"Paper {i}  >>"
                f"Title: {paper.title}\n"
                f"Abstract: {paper.content}\n"
                f"DOI: {paper.metadata['doi']}\n"
            )
            input_texts.append(paper_info)
            dois.append(paper.metadata['doi'])
            titles = titles + paper.title+" and"

        papers_text = "\n\n".join(input_texts)

        system_prompt = (
            f"{papers_text}\n\n"
            f"Start by explaining general concept of '{query}' the first sentence"
            f"The rest is describing how {top_k} paper ({titles[:-3]}) contributes to understanding {query}:\n\n"
            f"One-Paragraph for general concept of '{query}' and summary of ({titles[:-3]}) :"
            
        )

        print(system_prompt)
        result = self.summarizer(
            system_prompt, 
            max_new_tokens=300,  # More tokens for comprehensive single paragraph
            do_sample=True,
            temperature=0.5,  # Higher temperature for better flow
            pad_token_id=self.summarizer.tokenizer.eos_token_id,
            truncation=True,
            repetition_penalty=1.1  # Prevent repetition of input text
        )
        
        generated_text = result[0]['generated_text']
        summary_part = generated_text[len(system_prompt):].strip()
        
        summary_part = summary_part.replace('\n\n', ' ').replace('\n', ' ').strip()
        
        formatted_output = f"**Summary** {summary_part}\n\n**Citation**\n"
        for i, doi in enumerate(dois):
            formatted_output += f"[{i}] {doi}\n"
        
        return formatted_output.strip()