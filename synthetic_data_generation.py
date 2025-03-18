"""Code here is based off of the excellent synthetic data generating notebook from Jason Liu"""

import asyncio
from pydantic import BaseModel
import instructor
from openai import AsyncOpenAI
import random
from typing import List
import instructor
from asyncio import Semaphore
from tenacity import retry, stop_after_attempt, wait_fixed
from rich import print
import json
from tqdm.asyncio import tqdm

class DatasetStatement(BaseModel):
    statement_id: int
    statement: str 

class SpaceNeedleStatement(BaseModel):
    chain_of_thought: str
    statement: str

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def generate_space_needle_text(prompt_templates: List[str], aspects: List[str], personas: List[str], document_types: List[str], id: int, sem: Semaphore, client) -> DatasetStatement:
    async with sem:    
        prompt_template = random.choice(prompt_templates)
        aspect = random.choice(aspects) if "{aspect}" in prompt_template else None
        persona = random.choice(personas) if "{persona}" in prompt_template else None
        document_type = random.choice(document_types) if "{document_type}" in prompt_template else None

        formatted_prompt = prompt_template.format(
            aspect=aspect if aspect else "", #just in case we add in a template without aspect in the future
            persona=persona if persona else "",
            document_type=document_type if document_type else ""
        )
        
        # Generate the sentences
        response = await client.chat.completions.create(
            model="google/gemma-3-12b-it",
            messages=[
                {"role": "system", "content": "You are an assistant that generates varied, natural text about landmarks. Always provide responses directly without any preamble, introduction, or explanation. Never add phrases like 'Here is', 'Sure', or 'Certainly'. Deliver only the exact content requested."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=1.,
            response_model=SpaceNeedleStatement,
        )
        
        return DatasetStatement(
            statement_id=id,
            statement=response.statement
        )

async def create_synthetic_dataset(desired_samples: int) -> List[DatasetStatement]:
    """Creates a list of synthetic DatasetStatement objects about the Space Needle"""    
    client = instructor.from_openai(AsyncOpenAI(
        base_url = 'http://localhost:8000/v1', #vllm
        api_key=':)', #vllm 
    ))

    prompt_templates = [
    "Write one sentence about the Space Needle's {aspect} without any introduction. The sentence should be factual and directly address the topic.",
    "Generate exactly one sentence that mentions the Space Needle in the context of {aspect}. Start with the sentence itself, no preamble.",
    "Provide a single sentence where the Space Needle is mentioned while discussing {aspect}. Deliver only the sentence with no introduction.",
    "Write only one sentence that a {persona} might say or write about the Space Needle, focusing on its {aspect}. Give just the sentence itself.",
    "Generate exactly one sentence about the Space Needle that might appear in a {document_type} about {aspect}. Provide only the sentence, nothing else."
    ]

    # Define various aspects to generate data about
    aspects = [
        "architecture and design", "height and dimensions", "observation deck views", 
        "history and construction", "role in Seattle's skyline", "comparison to other towers",
        "restaurants and dining options", "as a tourist attraction", "cultural significance",
        "engineering features", "lighting and nighttime appearance", "renovation history",
        "surrounding Seattle Center area", "in popular media", "as a photography subject"
    ]

    # Define personas for variety
    personas = [
        "tourist", "architectural critic", "Seattle resident", "historian", 
        "engineer", "photographer", "child visitor", "restaurant reviewer",
        "urban planner", "travel blogger", "structural engineer"
    ]

    # Document types for variety
    document_types = [
        "travel guide", "architectural journal", "history book", "blog post",
        "news article", "academic paper", "social media post", "tour brochure",
        "engineering textbook", "personal memoir about Seattle"
    ]

    
    dataset = []
    sem = asyncio.Semaphore(10)  # Limit concurrent requests
    
    tasks = []
    for sample_id in range(desired_samples):
        task = generate_space_needle_text(
            prompt_templates=prompt_templates,
            aspects=aspects,
            personas=personas, 
            document_types=document_types,
            id=sample_id,
            sem=sem,
            client=client
        )
        tasks.append(task)
    
    # Use tqdm to show progress bar
    dataset = await tqdm.gather(*tasks)
    
    return dataset

async def save_to_json(dataset: List[DatasetStatement], filename: str) -> None:
    """Saves DatasetStatement objects to a JSON file"""
    # Convert to dictionary format
    data_dict = {item.statement_id: item.statement for item in dataset}
    
    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

async def main():
    space_needle_dataset = await create_synthetic_dataset(desired_samples=10000)
    await save_to_json(space_needle_dataset, "space_needle_dataset.json")

    print(f"Generated {len(space_needle_dataset)} unique sentences about the Space Needle")


if __name__ == "__main__":
    asyncio.run(main())