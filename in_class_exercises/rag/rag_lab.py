import html2text
import time
import re
from transformers import pipeline, AutoTokenizer, AutoModel
from torch import Tensor
import torch

# Each course description is a document

device='cpu'  # Change to 'cuda' if you have a GPU

print("INFO [] Loading CS course catalog")
# Load CS course catalog data
cs = open("cs.html", "rt").readlines()
h = html2text.HTML2Text()
h.ignore_links = True
txt = html2text.html2text("\n".join(cs))
newtxt = re.sub(r'\]\([^\)]*\)', r']', txt).replace('[', '').replace(']', '')
courses = newtxt.split('### ')[1:]  # "1:" -- remove preamble "course"

# This is a small but usable encoder for RAG
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
index_model = AutoModel.from_pretrained("thenlper/gte-small").to(device)

# TODO: Create RAG document index.
# You might want to replace '\n' with ' ' before indexing (that's what I did).
print("INFO [] Indexing courses")
index = []
for i, course in enumerate(courses):
    print(i)
    print(course)
    encoded_input = tokenizer(course.replace('\n', ' '), return_tensors='pt')
    encoded_input = { k:v.to(device) for (k,v) in encoded_input.items() }  # Load into GPU for faster processing
    with torch.no_grad():
        output = index_model(**encoded_input, output_hidden_states=True)
    # Take last hidden state from the model as the index vector for each document
    # TODO: finish me...
# TODO: finish me...

# Sample query
def query (query_model, generator, K, question):
    # Fetch the K most relevant documents based on the question and the document index.
    # Append these documents as context to the LLM query.
    # Then answer the question with the LLM.
    encoded_input = tokenizer(question, return_tensors='pt')
    encoded_input = { k:v.to(device) for (k,v) in encoded_input.items() }  # Load into GPU for faster processing
    # TODO: finish me...
    # prompt = ...
    result = generator(prompt, max_new_tokens=100)[0]['generated_text']
    cleaned_output = result[len(prompt):].strip()  # Remove the question itself from the output
    return cleaned_output

K = 3  # Feel free to change this
query_model = index_model

# Load LLM -- it's not great but good enough for this lab
generator = pipeline('text-generation', model="facebook/opt-iml-1.3b", device=device)

# Load and answer all questions
questions = open("questions.txt", "rt").readlines()
answers = {}
for question in questions:
    answers[question] = query(query_model, generator, K, question)
