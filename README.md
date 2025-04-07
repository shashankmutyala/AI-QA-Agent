# 🧠 AI-QA-Agent: Website-Aware Question Answering System

AI-QA-Agent is a smart question-answering system designed to extract and answer queries from any documentation-based website. It uses web crawling, vector indexing, and transformer-based embeddings to retrieve accurate answers from the most relevant web pages.

---

## 🚀 Features

- 🔎 Crawls and parses content from documentation websites  
- 📚 Converts web content into vector embeddings for semantic search  
- 🧠 Retrieves accurate answers using transformer-based NLP models  
- ⚡ Fast and scalable vector search using FAISS  
- 🌐 Provides a user-friendly web interface  
- 🔧 Modular design with robust error handling  
- 🧪 Integrated testing support  

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/shashankmutyala/AI-QA-Agent.git
cd AI-QA-Agent
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Embedding Model
```bash
mkdir -p downloaded_model
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('downloaded_model')"
```

---

## 🛠️ Usage

### ✅ Command-Line Interface
```bash
python src/main.py --url "https://docs.python.org/3/" --query "What is a generator?"
```

### 🌐 Web Interface

Start the web application:
```bash
python -m src.web_app
```

Then open your browser and navigate to: [http://localhost:5000](http://localhost:5000)

#### Crawl & Index Website:
- Enter a documentation website URL  
- Click **"Crawl & Index"**  
- Wait for crawling and indexing to complete  

#### Ask Questions:
- Enter your question  
- Click **"Ask Question"**  
- View the answer with sources and confidence score  

---

## 📁 Project Structure

```
AI-QA-Agent/
├── src/
│   ├── crawling/         # HTML parsing and crawling logic
│   ├── indexing/         # FAISS indexing and vector storage
│   ├── nlp/              # Embeddings and QA model handling
│   ├── utils/            # Common utility functions and error handling
│   ├── main.py           # CLI entry point
│   ├── web_app.py        # Flask app
│   └── config.py         # App configuration
├── tests/                # Unit and integration tests
├── docs/                 # Project documentation
├── requirements.txt      # Python dependency file
├── Dockerfile            # Docker container setup
└── README.md             # Project overview and instructions
```

---

## ✅ Tech Stack

- Python 3.8+  
- Flask  
- BeautifulSoup4  
- FAISS (Facebook AI Similarity Search)  
- Sentence Transformers / OpenAI Embeddings  
- PyTest (for testing)  

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 📌 Roadmap

- [ ] Add sitemap and multi-page crawling  
- [ ] Improve UI/UX for web interface  
- [ ] Add multilingual support  
- [ ] Deploy on Hugging Face Spaces  

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!  
Feel free to open an issue or submit a pull request.

---

## 🙋‍♂️ FAQ

**Q1: Can I use this with any website?**  
Yes, but it works best with static content or documentation pages. JavaScript-heavy sites may need extra handling.

**Q2: Does it require OpenAI API?**  
By default, it uses Sentence Transformers locally, but can be configured to use OpenAI’s API for embeddings.

**Q3: Can it be deployed online?**  
Yes! It can be easily deployed using Docker, or on platforms like Hugging Face Spaces or Streamlit Cloud.

---

## 🌟 Star This Project

If you found this useful, don’t forget to ⭐ the repo and share it with others!

## 🌟 Demo for This Project




https://github.com/user-attachments/assets/a08cc205-1c5a-4c87-bbfd-b260c9a3fffc


---
