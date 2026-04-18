# Detecting Malaria Using Deep Learning

<p align="center">
  <img src="https://pngimage.net/wp-content/uploads/2018/06/malaria-in-png-1.png" alt="Malaria Detection Logo" width="150" height="150">
</p>

## Introduction

This machine learning web application uses a convolutional neural network to process segmented blood smear cell images and predict whether they are infected with malaria.

The dataset used for training is based on segmented thin blood smear images published through the NIH malaria screening research effort.

## Purpose

Malaria diagnosis can be difficult in places where the disease is less common and clinical familiarity is lower. This project aims to support faster and more consistent screening by applying deep learning to a binary image classification problem where accuracy and speed both matter.

## Technology Stack

- [Flask](https://github.com/pallets/flask)
- [HTML](https://www.w3.org/TR/html52/)
- [CSS](https://developer.mozilla.org/en-US/docs/Web/CSS)
- [Bootstrap](https://getbootstrap.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

## Local Installation

1. Clone the repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the app with `python app.py`.
4. Open `http://127.0.0.1:5000/` in your browser.

## Render Deployment

This repo includes a minimal `render.yaml` blueprint for Render.

1. Push the project to GitHub.
2. In Render, create a new Blueprint or Web Service from this repo.
3. Fill in the required environment variables for `APP_BASE_URL`, `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `SMTP_USER`, `SMTP_PASSWORD`, and `SMTP_FROM_EMAIL`.
4. If you deploy on Railway, prefer setting `MYSQL_URL` from the MySQL service connection panel. The app now understands Railway-style URLs like `mysql://user:password@host:port/database` and will use the database name from that URL.
5. After the first deploy, set `APP_BASE_URL` to your public service URL, such as `https://your-service-name.onrender.com`.
6. If you are not using `MYSQL_URL`, use your MySQL server details for the `MYSQL_*` variables.

Notes:

- The app exposes a health check at `/healthz`.
- Profile images can be stored on Render's persistent disk with `PROFILE_UPLOAD_DIR=/var/data/profile_pics`.
- Python is pinned to `3.10.13` for TensorFlow `2.10.0` compatibility.

## CI/CD

- GitHub Actions runs tests on every push and pull request to `main` via [`.github/workflows/ci.yml`](C:\Malria\Malaria-Detection\.github\workflows\ci.yml).
- Automatic deployment is handled by [`.github/workflows/deploy.yml`](C:\Malria\Malaria-Detection\.github\workflows\deploy.yml) after the `CI` workflow succeeds, as long as you add a GitHub secret named `RENDER_DEPLOY_HOOK_URL`.
- In Render, you can also keep Git-based auto deploy enabled for the service if you prefer Render to redeploy directly from `main`.
