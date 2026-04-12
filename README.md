# utsa-gpt-api
Backend for UTSAGPT Senior Design Project

mainCL.py is used to run the backend completely through the command line

main.py holds the logic for FastAPI and endpoints

Make sure you've updated your .env with the proper files and you have the correct .json for firebase admin priv

INSTRUCTIONS TO RUN FASTAPI:
1. go to utsa-gpt-api/
2. run "cd backend"
3. run "uvicorn main:app --reload
4. Wait until you see "INFO     Application startup complete." this can take a minute or two
5. You can check automatic FastAPI docs via "http://localhost:8000/docs"
6. You can test endpoints docs by opening a second powershell instance and pasting (one line):
Invoke-RestMethod -Uri "http://localhost:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question": "Who plays on Sunday?"}'
