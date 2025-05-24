from typing import List
import aiohttp
import asyncio
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
            # Create a timeout for the client session
            timeout = aiohttp.ClientTimeout(total=90)  # 90 seconds timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Add retry logic for API calls
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"DeepSeek API returned status code {response.status}: {error_text}")
                            
                            response_data = await response.json()
                            summary_part = response_data["choices"][0]["message"]["content"].strip()
                            break  # Success, exit the retry loop
                    
                    except (aiohttp.ClientConnectorError, aiohttp.ClientResponseError, aiohttp.ServerTimeoutError) as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise Exception(f"Failed to connect to DeepSeek API after {max_retries} attempts: {str(e)}")
                        print(f"Connection error, retrying ({retry_count}/{max_retries}): {str(e)}")
                        await asyncio.sleep(2 * retry_count)  # Exponential backoff
            
            # Format the final output with citations
            formatted_output = f"**Summary on '{query}'**\n{summary_part}\n\n**Citations**\n"
            for i, paper_id in enumerate(paper_ids):
                formatted_output += f"[{i}] {paper_id}\n"
            
            return formatted_output.strip()
            
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    async def generate_insights(self, paper: 'SearchResult') -> str:
        """
        Generate AI insights for a specific paper using its title and abstract
        """
        system_prompt = (
            "You are an expert academic researcher specializing in scientific paper analysis. "
            "Your task is to provide comprehensive insights about a research paper."
        )
        
        user_prompt = (
            f"Title: {paper.title}\n"
            f"Abstract: {paper.content}\n\n"
            f"Please provide a comprehensive analysis of this research paper with the following structure:\n\n"
            f"1. **Executive Summary** (2-3 sentences): Provide a clear, concise overview of what this paper is about and its main contribution.\n\n"
            f"2. **Key Findings** (3-5 bullet points): List the most important discoveries, results, or conclusions from this research.\n\n"
            f"3. **Methodology** (2-3 sentences): Explain the approach, methods, or techniques used in this study.\n\n"
            f"4. **Significance & Impact** (2-3 sentences): Discuss why this research matters, its potential applications, or its contribution to the field.\n\n"
            f"5. **Technical Innovation** (if applicable): Highlight any novel techniques, algorithms, or approaches introduced.\n\n"
            f"Please be specific and focus on the actual content rather than general statements. "
            f"Use clear, accessible language while maintaining technical accuracy."
        )

        print(f"Generating insights for paper: {paper.title}")
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000  # Slightly higher token limit for detailed insights
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=90)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"DeepSeek API returned status code {response.status}: {error_text}")
                            
                            response_data = await response.json()
                            insights = response_data["choices"][0]["message"]["content"].strip()
                            return insights
                    
                    except (aiohttp.ClientConnectorError, aiohttp.ClientResponseError, aiohttp.ServerTimeoutError) as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise Exception(f"Failed to connect to DeepSeek API after {max_retries} attempts: {str(e)}")
                        print(f"Connection error, retrying ({retry_count}/{max_retries}): {str(e)}")
                        await asyncio.sleep(2 * retry_count)
            
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            return f"Error generating insights: {str(e)}"