# Deploy AgriNathi to Render (quick guide)

This file explains how to deploy this repository to Render using GitHub auto-deploy.

Prerequisites
- A GitHub repository with this project pushed.
- A Render account (https://render.com) with access to the GitHub repo.

Steps

1) Push code to GitHub

   - Initialize git (if you haven't) and push to a GitHub repository:

```powershell
git init
git add .
git commit -m "Initial commit for Render deployment"
git branch -M main
git remote add origin https://github.com/<your-org-or-username>/<repo>.git
git push -u origin main
```

2) Create a new Web Service on Render

- Go to https://dashboard.render.com/new and choose "Web Service".
- Connect your GitHub account and pick the repository.
- Branch: `main` (or your deployment branch)
- Build Command: `pip install -r requirements.txt` (provided in `render.yaml`)
- Start Command: `gunicorn run:app --workers 3 --bind 0.0.0.0:$PORT` (also provided in `render.yaml` and `Procfile`)
- Environment: `Python` (Render will detect from repo)

3) Environment variables and secrets

- Add any keys (Google credentials, Kaggle, API keys) in the Render dashboard under the service settings -> Environment -> Environment Variables.
- Do NOT commit credentials to the repository. Use Render's secure ENV feature.

4) Static files and uploads

- If your app stores uploaded files locally (`data/collected_images/`), note that Render's filesystem is ephemeral; files written to disk will be lost on deploy or instance restart. For production, use external storage (S3, Azure Blob, or Render's Persistent Disks).

5) Verify the deployment

- After creating the service Render will build and deploy. Monitor the build logs on Render.
- When the deployment finishes, visit the provided URL.

Troubleshooting

- If build fails due to large packages (TensorFlow), consider using a Docker service on Render with a prebuilt image or use a smaller CPU-optimized TF (or run heavy ML tasks on separate worker instances).
- If you need GPU for training, Render does not provide GPU instances on the free plan; use a dedicated GPU provider or GCP/Azure/AWS GPU VM.

Optional: render.yaml

You can use the included `render.yaml` to configure the service via code. When creating a new service choose the "from a render.yaml" option and point to the file.

That's it — when connected, Render will auto-deploy on pushes to the selected branch.
