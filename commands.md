ollama serve
ollama list

lsof -i :11434
lsof -i :5000

kill 592

curl -X POST http://localhost:5001/query -H "Content-Type: application/json" -d '{"prompt": "In 10 words describe AI"}'

curl -F "file=@paper_mini.pdf" http://localhost:5001/upload

curl -X POST http://localhost:5001/query -H "Content-Type: application/json" -d '{"prompt": "What is the main theme of this document?", "pdf_id": 1}'
