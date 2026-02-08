# Task: Implement Bonus Features

Remaining features to reach "Pro" status for the RAG Chatbot.

## 1. "Remember This" (Explicit Memory)

- [ ] Add `UserMemory` table in `app/models.py`.
- [ ] Add manual fact extraction in `app/api.py` (check for "remember this:" prefix).
- [ ] Inject `user_memories` into the prompt template in `app/rag/qa.py`.

## 2. Conversation Search

- [ ] Add `GET /conversations/search` endpoint in `app/api.py`.
- [ ] Implement SQL search across titles and message contents.
- [ ] Add search bar UI in the History drawer in `index.html`.

## 3. Conversation Analytics

- [ ] Add `GET /analytics` endpoint.
- [ ] Calculate conversation stats (counts, avg length).
- [ ] Add a simple dashboard or modal in the UI.

## 4. Final Polish

- [ ] Update `ai_developer_learning_path.md` once complete.
