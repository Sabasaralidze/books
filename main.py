import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

def load_api_key():
    """Load OpenAI API key from .env or prompt user if not found."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Prompt user for key at runtime
        api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env or enter it when prompted.")
    openai.api_key = api_key
    # Optionally set custom base (e.g., Groq)
    groq_base = os.getenv("GROQ_API_BASE")
    if groq_base:
        openai.api_base = groq_base

def load_book_data(csv_path: str) -> pd.DataFrame:
    """Load and validate book dataset CSV."""
    df = pd.read_csv(csv_path)
    expected = ["title", "author", "genre", "mood", "description", "keywords"]
    for col in expected:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataset. Expected columns: {expected}")
    if "year" not in df.columns:
        df["year"] = None
    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["author"].fillna("") + " " +
        df["genre"].fillna("") + " " +
        df["mood"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["keywords"].fillna("")
    )
    return df

def build_tfidf_matrix(text_series: pd.Series):
    """Build TF-IDF vectorizer and matrix from text series."""
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(text_series)
    return tfidf, matrix

def suggest_books(preference_input: str, tfidf: TfidfVectorizer, book_matrix, data: pd.DataFrame, count=3) -> pd.DataFrame:
    """Return top matching books based on preference input."""
    if not preference_input.strip():
        return pd.DataFrame([{
            "title": "No input provided",
            "author": "",
            "description": "Please provide some preferences first.",
            "year": None
        }])
    query_vector = tfidf.transform([preference_input])
    scores = cosine_similarity(query_vector, book_matrix).flatten()
    top_idxs = scores.argsort()[-count:][::-1]
    return data.iloc[top_idxs][["title", "author", "description", "year"]]

def chat_response(chat_log, model_name="gpt-3.5-turbo") -> str:
    """Generate chat response using OpenAI API."""
    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=chat_log
        )
        return resp['choices'][0]['message']['content']
    except Exception as e:
        return f"[Error generating response: {e}]"

def book_bot(csv_path="books_dataset.csv", model_name="gpt-3.5-turbo"):
    # Load API key
    try:
        load_api_key()
    except Exception as e:
        print(f"Error loading API key: {e}")
        return
    # Load data
    try:
        data = load_book_data(csv_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    tfidf, book_matrix = build_tfidf_matrix(data["combined_text"])

    print("\n⏰ Hello! I'm your book-finding assistant.")
    print("Chat with me about your preferences. Type 'recommend' when ready, or 'exit' to quit.\n")
    chat_memory = [
        {"role": "system", "content": "You are a conversational assistant that learns what kind of books a user likes."}
    ]
    preference_notes = ""

    while True:
        try:
            user_text = input("You: ")
        except EOFError:
            print("\nBot: Goodbye!")
            break
        if user_text is None:
            continue
        user_text = user_text.strip()
        if not user_text:
            continue
        if user_text.lower() == "exit":
            print("Bot: Take care! 💤")
            break
        elif user_text.lower() == "recommend":
            print("\nBot: Based on our discussion, I think you'd enjoy these titles:\n")
            results = suggest_books(preference_notes, tfidf, book_matrix, data)
            if results.empty:
                print("No recommendations available. Please tell me more about your preferences.")
            else:
                for _, row in results.iterrows():
                    title = row.get("title", "Unknown Title")
                    author = row.get("author", "Unknown Author")
                    year = row.get("year", "Unknown Year")
                    desc = row.get("description", "")
                    print(f"- {title} by {author} ({year})")
                    if desc:
                        print(f"  {desc}\n")
            continue
        # Accumulate preference notes and chat
        preference_notes += " " + user_text
        chat_memory.append({"role": "user", "content": user_text})
        reply = chat_response(chat_memory, model_name)
        chat_memory.append({"role": "assistant", "content": reply})
        print(f"Bot: {reply}\n")

if __name__ == "__main__":
    # Update CSV filename or model_name as needed
    book_bot(csv_path="books_dataset.csv", model_name="gpt-3.5-turbo")
