from typing import List
import aiohttp
import json
from app.models.search import SearchResult
from app.core.config import settings


class DeepSeekService:
    def __init__(self):
        self.api_key = settings.DEEPSEEK_KEY
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    async def summarize(
        self, query: str, context: List[SearchResult], top_k: int = 3
    ) -> str:
        
        input_texts = []
        paper_ids = []
        titles = []
        for i, paper in enumerate(context[:top_k]):
            # Get the paper ID directly from the paper object, not from metadata
            paper_id = paper.id if hasattr(paper, 'id') and paper.id else paper.metadata.get('id', f'unknown-{i}')
            paper_info = (
                f"Paper {i}  >>"
                f"Title: {paper.title}\n"
                f"Abstract: {paper.content}\n"
                f"ID: {paper_id}\n"
            )
            input_texts.append(paper_info)
            paper_ids.append(paper_id)
            titles.append(paper.title)

        papers_text = "\n\n".join(input_texts)
        titles_text = ", ".join([f"'{title}'" for title in titles])

        # Generate a clean, well-referenced bullet-point summary
        system_prompt = "You are an expert academic researcher that summarizes scientific papers accurately and concisely."
        user_prompt = (
            f"{papers_text}\n\n"
            f"Based on your query '{query}', these are the most relevant papers:\n\n"
            f"1. Begin with a brief introduction about '{query}'.\n"
            f"2. For EACH paper, create bullet points that explain:\n"
            f"   - The main methodology or approach used\n"
            f"   - Key findings and results\n"
            f"   - Specific contributions to the field\n"
            f"3. Every bullet point MUST reference the source paper by number [0], [1], etc.\n"
            f"4. Paraphrase rather than directly quote - explain concisely what each paper says about '{query}'.\n"
            f"5. Focus only on information relevant to '{query}'.\n\n"
            f"Summary of '{query}' based on papers: {titles_text}:"
        )

        print(f"Sending request to DeepSeek API with query: {query}")
        
        # Create the API request payload
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"DeepSeek API returned status code {response.status}: {error_text}")
                    
                    response_data = await response.json()
                    summary_part = response_data["choices"][0]["message"]["content"].strip()
                    
            # Format the final output with citations
            formatted_output = f"**Summary on '{query}'**\n{summary_part}\n\n**Citations**\n"
            for i, paper_id in enumerate(paper_ids):
                formatted_output += f"[{i}] {paper_id}\n"
            
            return formatted_output.strip()
            
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            return f"Error generating summary: {str(e)}"