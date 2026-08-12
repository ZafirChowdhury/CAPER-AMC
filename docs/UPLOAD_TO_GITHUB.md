# Upload CAPER-AMC to GitHub

## Option 1 — GitHub website

1. Create a new repository named `CAPER-AMC`.
2. Do **not** initialize it with another README if you want to use the supplied README directly.
3. Open the new repository and choose **Add file → Upload files**.
4. Upload the contents of the `CAPER-AMC` folder.
5. Commit the files.

The included `.pth` checkpoint is below GitHub's normal 100 MB per-file limit.

## Option 2 — Terminal

From inside the `CAPER-AMC` folder:

```bash
git init
git add .
git commit -m "Initial CAPER-AMC release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CAPER-AMC.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## Important

Do not upload the raw `RML2016.10a_dict.pkl` dataset to this repository. It is already excluded by `.gitignore`.
