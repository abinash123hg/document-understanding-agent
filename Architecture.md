# Architecture

## Main components

- `backend/`: API, authentication, and document processing.
- `frontend/`: browser interface.
- `data/`: document data and retrieval indexes.
- `.venv/`: local Python environment; do not commit it.

## Runtime flow

1. User opens the frontend.
2. User uploads an allowed document.
3. Backend validates the file.
4. Document text is extracted and processed.
5. Processed content is stored securely.
6. User asks a question.
7. Relevant content is retrieved.
8. The language model generates the answer.
9. The response is returned to the user.

## Security boundaries

- Authentication before protected operations.
- Authorization before document or admin access.
- Secrets loaded from environment variables.
- Uploaded files stored securely.
- Errors must not expose sensitive internal details.