# spinaki.github.io — Jekyll + Minimal Mistakes starter

Pinaki's ML / GenAI Focused Website 

This is a GitHub Pages blog using:

- Jekyll
- Minimal Mistakes (dark skin)
- MathJax for LaTeX
- Mermaid for diagrams
- Lunr.js search
- Sticky table of contents
- Copy buttons on code blocks

## Local dev

```bash
bundle install
bundle exec jekyll serve
```

Then open http://localhost:4000 in your browser.

## Deploy

Push to the `main` branch of the `spinaki.github.io` repo.  
GitHub Actions workflow `.github/workflows/jekyll.yml` will build and deploy to GitHub Pages.
