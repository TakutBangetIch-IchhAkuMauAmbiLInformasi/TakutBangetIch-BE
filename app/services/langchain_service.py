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
        for i, paper in enumerate(context[:top_k]):
            paper_info = (
                f"Title: {paper.title}\n"
                f"Abstract: {paper.content}\n"
                f"DOI: {paper.metadata['doi']}\n"
            )
            input_texts.append(paper_info)
            dois.append(paper.metadata['doi'])

        papers_text = "\n\n".join(input_texts)

        # Generate a single paragraph summary combining all papers
        system_prompt = (
            f"Write one paragraph summarizing these research papers about '{query}'. "
            f"Start by explaining what '{query}' refers to, then describe how each study contributes to understanding this concept:\n\n"
            f"{papers_text}\n\n"
            f"Combined One-Paragraph Summary:"
        )

        # Generate with parameters optimized for coherent single paragraph
        result = self.summarizer(
            system_prompt, 
            max_new_tokens=250,  # More tokens for comprehensive single paragraph
            do_sample=True,
            temperature=0.5,  # Higher temperature for better flow
            pad_token_id=self.summarizer.tokenizer.eos_token_id,
            truncation=True,
            repetition_penalty=1.1  # Prevent repetition of input text
        )
        
        # Extract and clean the generated text
        generated_text = result[0]['generated_text']
        summary_part = generated_text[len(system_prompt):].strip()
        
        # Clean up any artifacts and ensure single paragraph
        summary_part = summary_part.replace('\n\n', ' ').replace('\n', ' ').strip()
        
        # Format the final output as single paragraph
        formatted_output = f"**Summary** {summary_part}\n\n**Citation**\n"
        for i, doi in enumerate(dois):
            formatted_output += f"[{i}] {doi}\n"
        
        return formatted_output.strip()