# AI Debate App

An interactive debate application where AI agents argue for and against a given topic using the Groq LLM.

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Running Locally

Run the Streamlit app:
```bash
streamlit run main.py
```

## Deployment

This app is configured for deployment on Streamlit Cloud. To deploy:

1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select this repository
6. Set the main file path as `main.py`
7. Add your `GROQ_API_KEY` in the secrets management section
8. Deploy!

## Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required) 