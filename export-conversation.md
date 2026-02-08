# Task: Implement Conversation Export

Allows users to download their chat history as a formatted text file.

## 1. Backend Implementation

- [x] Add `GET /conversations/{conv_id}/export` endpoint in `app/api.py`.
- [x] Format the message history into a readable string (e.g., Markdown).
- [x] Return the content with `Content-Type: text/plain` and `Content-Disposition: attachment`.

## 2. Frontend Implementation

- [x] Add an "Export" button in the chat interface.
- [x] Connect the button to the backend export endpoint.
- [x] Handle the file download in the browser.

## 3. Verification

- [x] Chat with the bot.
- [x] Click Export.
- [x] Verify the downloaded file contains the correct history.
