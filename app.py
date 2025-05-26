from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# create structure for body request
class EssayRequest(BaseModel):
    question: str
    score: str

checkpoint_path = "./gpt_ielts/checkpoint-8170"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Load the model
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)

app = FastAPI()

def generate_ielts_essay(question, overall, max_length=512):
    model.eval()
    input_text = f"From the topic: {question}, please write an IELTS essay that can achieve band {overall}:\n Essay: \n"
    # print(input_text)
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Generate the essay using the model
    output_ids = model.generate(
        input_ids,
        eos_token_id=tokenizer.convert_tokens_to_ids('[END]'),
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=3,  # To prevent repetition
        early_stopping=True,
        temperature=0.3,  # For controlled creativity
        top_p=0.85,        # Top-p sampling
        top_k=50,         # Top-k sampling
        do_sample=True    # Enable sampling
    )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


@app.post("/generate")
def generate_essay(request: EssayRequest):
    essay = generate_ielts_essay(request.question, request.score)
    return {"essay": essay}