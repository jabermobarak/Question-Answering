from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json 
    context = data.get('context', '')
    question = data.get('question', '')
    
    if not context or not question:
        return jsonify({'error': 'Context or question missing'}), 400
    
    result = qa_pipeline(question=question, context=context)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
