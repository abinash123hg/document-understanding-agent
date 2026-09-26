# Engineering and Security Rules

- Never hardcode API keys, passwords, tokens, or private paths.
- Use environment variables for secrets.
- Never commit `.env`, `.venv`, uploads, or generated secrets.
- Validate every user input.
- Restrict file types and file sizes.
- Generate server-side filenames.
- Require authentication for protected endpoints.
- Check permissions for every document and admin operation.
- Do not return stack traces to users.
- Do not log passwords, tokens, API keys, or document contents.
- Apply rate limits to login, upload, and chat endpoints.
- Use HTTPS in deployed environments.
- Verify webhook signatures.
- Do not trust client-provided payment status.
