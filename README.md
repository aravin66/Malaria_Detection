# Detecting Malaria using Deep Learning 🦟🦠

<p align="center">
  <a href="https://github.com/HarshCasper/Malaria-Detection">
    <img src="https://pngimage.net/wp-content/uploads/2018/06/malaria-in-png-1.png" alt="Logo" width="150" height="150">
  </a>
  
Contributor's Hack 2020 is a program that helps students grow with **OPEN SOURCE**. This initiative by **HakinCodes** provides you the best platform to improve your skills and abilities by contributing to vast variety of OPEN SOURCE Projects and opportunity to interact with the mentors and the Organizing Team.

<p align="center">
  <a href="https://hakincodes.tech/">
    <img src="https://user-images.githubusercontent.com/54139847/87952512-882a5600-cac7-11ea-939d-8304a641d8a9.png" alt="HakinCodes">
  </a>
</p>

## 📌 Introduction

This Machine Learning Web Application utilizes a Two-Layered Convolutional Neural Network to process the Cell Images and predict if they are Malarial with an accuracy of nearly 95%. The [Dataset](https://www.dropbox.com/s/f20w7sqvxvl0p68/malaria-dataset.zip) to process the Deep Learning Algorithm is taken from the official US National Library of Medicine's NIH Website which is a repository of segmented cells from the thin blood smear slide images from the Malaria Screener research activity.

## 🎯 Purpose of the Project

Where malaria is not endemic any more (such as in the United States), health-care providers may not be familiar with the disease. Clinicians seeing a malaria patient may forget to consider malaria among the potential diagnoses and not order the needed diagnostic tests. Laboratorians may lack experience with malaria and fail to detect parasites when examining blood smears under the microscope. Malaria is an acute febrile illness. 

In a non-immune individual, symptoms usually appear 10–15 days after the infective mosquito bite. The first symptoms – fever, headache, and chills – may be mild and difficult to recognize as malaria. If not treated within 24 hours, P. falciparum malaria can progress to severe illness, often leading to death. 

This Project aims to provides a handy tool to utilize the power of Machine Learning and Artificial Intelligence in Binary Classification Problems where time and accuracy is the paramount objective of classification.

## 🏁 Technology Stack

* [Flask](https://github.com/pallets/flask)
* [HTML](https://www.w3.org/TR/html52/)
* [CSS](https://developer.mozilla.org/en-US/docs/Web/CSS)
* [Bootstrap](https://getbootstrap.com/)
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](http://keras.io/)

## 🏃‍♂️ Local Installation

1. Drop a ⭐ on the Github Repository. 
2. Clone the Repo by going to your local Git Client and pushing in the command: 

```sh
https://github.com/HarshCasper/Malaria-Detection.git
```

3. Install the Packages: 
```sh
pip install -r requirements.txt
```

4. At last, push in the command:
```sh
python app.py
```

5. Go to ` http://127.0.0.1:5000/` and enjoy the application.

## 📜 LICENSE

[MIT](https://github.com/HakinCodes/Malaria-Detection/blob/master/LICENSE)

## Render Deployment

This repo now includes a minimal `render.yaml` blueprint for Render.

1. Push the project to GitHub.
2. In Render, create a new Blueprint or Web Service from this repo.
3. Fill in the prompted environment variables for `APP_BASE_URL`, `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `SMTP_USER`, `SMTP_PASSWORD`, and `SMTP_FROM_EMAIL`.
4. After the first deploy, set `APP_BASE_URL` to your Render URL, such as `https://your-service-name.onrender.com`.
5. Use your MySQL server details for the `MYSQL_*` variables.

Notes:

- The app exposes a health check at `/healthz`.
- Profile images can be stored on Render's persistent disk with `PROFILE_UPLOAD_DIR=/var/data/profile_pics`.
- Python is pinned to `3.10.13` for TensorFlow `2.10.0` compatibility.
