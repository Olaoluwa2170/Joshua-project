# Deploying the Maize Yield Dashboard online

The dashboard is a standard Streamlit app, so it can be hosted for free. The
easiest route is **Streamlit Community Cloud**. Two alternatives (Hugging Face
Spaces, Render) are noted at the end.

> **Note:** deployment requires signing in to a hosting service with your own
> account, so these are steps for *you* to run — they cannot be done on your behalf.

---

## Option A — Streamlit Community Cloud (recommended, free)

**What you need:** a GitHub account and a Streamlit account (sign in with GitHub).

1. **Put the project on GitHub.**
   From the project root:
   ```bash
   git init
   git add .
   git commit -m "Maize yield prediction: models + dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
   Make sure the committed repo includes `outputs/models/` (the `.joblib` files
   and `model_metadata.json`) — the dashboard loads the models from there. The
   Random Forest model is ~15 MB, which is fine for GitHub.

2. **Deploy.**
   - Go to <https://share.streamlit.io> and sign in with GitHub.
   - Click **Create app** → **Deploy a public app from GitHub**.
   - Repository: your repo. Branch: `main`. **Main file path:** `dashboard/app.py`.
   - Click **Deploy**. First build takes a few minutes while it installs
     `requirements.txt`.

3. **Done.** You get a public URL like
   `https://<your-app>.streamlit.app` that you can share or show in your defence.

**Why it just works:** `requirements.txt` pins the exact library versions used to
train the models, and `app.py` loads everything with paths relative to the repo,
so no configuration is needed.

---

## Option B — Hugging Face Spaces (also free)
1. Create a new **Space** at <https://huggingface.co/spaces> → SDK: **Streamlit**.
2. Upload the project files (or push with git).
3. Rename `dashboard/app.py` to `app.py` at the repo root **or** set the app file
   in the Space settings. Keep `requirements.txt` at the root.

## Option C — Render (free web service)
1. New → **Web Service**, connect the GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command:
   `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`

---

## Before you deploy — quick checklist
- [ ] `outputs/models/` (the four `.joblib` files + `model_metadata.json`) is committed.
- [ ] `requirements.txt` is at the project root.
- [ ] The app runs locally with `streamlit run dashboard/app.py`.
- [ ] `predictions.db` is **not** required — it is created automatically on the host.
