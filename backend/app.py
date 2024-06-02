from flask import Flask, jsonify, request 
import cohere
from pinecone import Pinecone
import os 
from flask_cors import CORS, cross_origin

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
cors = CORS(app, resources={r'*': {'origins':'*'}})

pinecone_api_key = os.getenv('p_api_key')
cohere_api_key = os.getenv('c_api_key')

@app.route("/")
@cross_origin(origins='*')
def hello_world():
    return "<p>Hello, World!</p>"

co = cohere.Client(cohere_api_key)
pc = Pinecone(pinecone_api_key) # add your pinecone API key here

index_name = 'entrez'
index = pc.Index(index_name)

limit = 3000

def retrieve(query):
    xq = co.embed(
        texts=[query],
        model='multilingual-22-12',
        truncate='NONE'
    ).embeddings

    # Search pinecone index for context passage with the answer
    xc = index.query(vector=xq, top_k=3, include_metadata=True)

    # Extract relevant information from the matches
    titles = [str(x['metadata']['Title']) for x in xc['matches']]
    abstracts = [str(x['metadata']['Abstract']) for x in xc['matches']]
    authors = [str(x['metadata']['Authors']) for x in xc['matches']]
    publication_years = [str(x['metadata']['Publication Year']) for x in xc['matches']]

    # Combine the information into formatted contexts
    contexts = [
        f"Title: {title}\nAbstract: {abstract}\nAuthors: {author}\nPublication Year: {publication_year}\n"
        for title, abstract, author, publication_year in zip(titles, abstracts, authors, publication_years)
    ]

    # Build the prompt with the retrieved contexts included
    prompt_start = (
        "Answer the Query based on the contexts.\n\nContext:\n"
    )
    prompt_end = (
        f"\n\nQuery:\n\nPlease provide the summary along with {titles} & {authors} at the end when the exercises have been suggested based on {query}.\n"
    )

    # Append contexts until hitting the limit
    combined_contexts = ""
    for context in contexts:
        if len(combined_contexts + context) > limit:
            break
        combined_contexts += context

    prompt = prompt_start + combined_contexts + prompt_end
    return prompt


def complete(prompt):
  response = co.generate(
                          model='command',
                          prompt=prompt,
                          max_tokens=3000000,
                          temperature=0.4,
                          k=0,
                          stop_sequences=['\n\n'],
                          return_likelihoods='NONE'
                        )
  return response.generations[0].text.strip()

@app.route('/api/predict', methods=['POST'])
def predict():
    query = request.json.get("query")
    query_with_contexts = retrieve(query)
    print(query_with_contexts)
    bot = complete(query_with_contexts)

    return {"bot": bot}
    
if __name__ == '__main__':
    app.run(debug=True)