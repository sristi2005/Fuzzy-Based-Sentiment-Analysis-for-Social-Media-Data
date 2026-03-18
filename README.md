# Fuzzy-Based Sentiment Analysis for Social Media Data

A full-stack **NLP web application** that analyzes social media text using a **hybrid sentiment analysis approach** combining **VADER**, **TextBlob**, and a **Fuzzy Inference System (FIS)**. The system produces a **nuanced sentiment score (1–100)** and a final sentiment label (Positive, Neutral, Negative).

The project demonstrates how **fuzzy logic can improve traditional sentiment analysis by resolving conflicting sentiment signals from multiple models.**

---

# Project Demo

Example Input:

```
"I absolutely love this new design! It's beautiful and fast, though slightly expensive."
```

Output:

```
Sentiment Score: 82
Label: Positive
```

The system intelligently combines **VADER compound scores** and **TextBlob polarity scores** using fuzzy logic rules.

---

# Features

• Hybrid NLP sentiment analysis using **VADER + TextBlob**
• **Fuzzy Inference System** for sentiment decision making
• Real-time sentiment analysis via **FastAPI backend**
• Modern **glassmorphism UI design**
• Animated sentiment progress bar/gauge
• Handles **mixed sentiment statements intelligently**
• Lightweight and fast API processing

---

# System Architecture

```
User Input (Social Media Text)
        │
        ▼
Frontend (HTML + CSS + JS)
        │
        ▼
FastAPI Backend
        │
        ▼
NLP Sentiment Extraction
(VADER + TextBlob)
        │
        ▼
Fuzzy Inference System
(scikit-fuzzy)
        │
        ▼
Final Sentiment Score (1–100)
+ Sentiment Label
```

---

# Tech Stack

## Backend

* Python
* FastAPI
* NLTK (VADER Sentiment Analyzer)
* TextBlob
* Scikit-Fuzzy
* NumPy
* Uvicorn

## Frontend

* HTML
* CSS (Glassmorphism design)
* JavaScript

---

# Project Structure

```
fuzzy-sentiment-analysis/
│
├── backend/
│   ├── main.py
│   ├── nlp_engine.py
│   ├── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│
└── README.md
```

---

# Installation

Clone the repository

```
git clone https://github.com/yourusername/fuzzy-sentiment-analysis.git
cd fuzzy-sentiment-analysis
```

Create virtual environment

```
python -m venv venv
```

Activate virtual environment

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

Install dependencies

```
pip install -r backend/requirements.txt
```

---

# Running the Application

Start the FastAPI server

```
cd backend
uvicorn main:app --reload --port 8000
```

Open in browser

```
http://127.0.0.1:8000
```

---

# Example Test Cases

Positive Example

```
"I absolutely love this product. It works perfectly!"
```

Negative Example

```
"This is the worst experience I have ever had."
```

Mixed Sentiment Example

```
"The design is beautiful but the performance is slow."
```

The fuzzy logic engine helps **resolve conflicting sentiment signals**.

---

# How the Fuzzy Logic Works

Inputs:

* **VADER Compound Score (-1 to 1)**
* **TextBlob Polarity (-1 to 1)**

Fuzzy Variables:

* Negative
* Neutral
* Positive

Example Rule:

```
IF VADER is Positive AND TextBlob is Positive
THEN Sentiment is Positive

IF VADER is Negative AND TextBlob is Neutral
THEN Sentiment is Negative
```

The fuzzy inference system converts these rules into a **final crisp sentiment score (1–100).**

---

# Future Improvements

• Support for **Twitter API / Reddit data**
• Add **deep learning sentiment models (BERT)**
• Store analyzed text in a **database**
• Add **sentiment trend analytics dashboard**
• Deploy on **AWS / Docker**
