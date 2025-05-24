from typing import List
from huggingface_hub import InferenceClient
from app.models.search import SearchResult
from app.core.config import settings


class LangchainService:
    def __init__(self):
        self.client = InferenceClient(
            model="HuggingFaceH4/zephyr-7b-beta",  
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
                f"ID: {paper.id}\n"
            )
            input_texts.append(paper_info)
            dois.append(paper.id)
            titles = titles + paper.title+" and"

        papers_text = "\n\n".join(input_texts)

        system_prompt = (
            f"<|system|>\n"
            f"You are a scientific research assistant. Provide a comprehensive summary based on the given papers.\n"
            f"<|user|>\n"
            f"Research Papers:\n{papers_text}\n\n"
            f"Task: Write a one-paragraph summary that:\n"
            f"1. First explains the general concept of '{query}'\n"
            f"2. Then describes how the {top_k} papers ({titles[:-3]}) contribute to understanding {query}\n\n"
            f"Please provide a cohesive paragraph summary:\n"
            f"<|assistant|>\n"
        )

        print(system_prompt)
        
        try:
            # Use Hugging Face Inference API
            response = self.client.text_generation(
                prompt=system_prompt,
                max_new_tokens=300,
                temperature=0.5,
                do_sample=True,
                repetition_penalty=1.1,
                return_full_text=False  # Only return generated text, not the prompt
            )
            
            # Extract the generated text
            if isinstance(response, str):
                summary_part = response.strip()
            else:
                # Handle different response formats
                summary_part = response.get('generated_text', '').strip()
            
        except Exception as e:
            print(f"Error calling Hugging Face API: {e}")
            return f"Error generating summary: {str(e)}"
        
        # Clean up the summary
        summary_part = summary_part.replace('\n\n', ' ').replace('\n', ' ').strip()
        
        # Format the final output
        formatted_output = f"**Summary** {summary_part}\n\n**Link**\n"
        for i, doi in enumerate(dois):
            formatted_output += f"[{i}] {doi}\n"
        
        return formatted_output.strip()