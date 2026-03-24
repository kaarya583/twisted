# TwistEd

Streamlit app for real-time NOAA alerts, safer route guidance, historical weather Q&A, and emergency education.

## Features

- Live severe weather alerts from NOAA
- Route risk comparison by state
- Historical-weather chatbot (facts and context)
- Learn section with disaster basics and emergency actions

## Repo

```text
twisted/
├── twisted.py
├── config.py
├── requirements.txt
├── test_twisted_app.py
├── Dockerfile
└── README.md
```

## Run

```bash
pip install -r requirements.txt
streamlit run twisted.py
```

Optional `.env`:

```env
OPENAI_API_KEY=your_key_here
```

## Test

```bash
python -m unittest -v test_twisted_app.py
```

## Note

Educational tool only. Follow official NOAA/NWS alerts and local emergency guidance.
