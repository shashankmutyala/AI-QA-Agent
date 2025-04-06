import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, AppConfig
from src.crawling.crawler import Crawler
from src.crawling.parser import Parser
from src.indexing.indexer import Indexer
from src.indexing.storage import FAISSStorage
from src.nlp.query_engine import QueryEngine
from src.nlp.formatter import ResponseFormatter
from src.utils.validators import is_valid_url, validate_query, check_url_accessibility

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create required directories
Path('data/storage').mkdir(parents=True, exist_ok=True)
Path('data/cache').mkdir(parents=True, exist_ok=True)

# Load config
config = load_config()

# Global variable to track the currently loaded domain
current_domain = None
query_engine = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/crawl', methods=['POST'])
def crawl_website():
    url = request.form.get('url')

    if not url:
        return jsonify({'status': 'error', 'message': 'URL is required'})

    if not is_valid_url(url):
        return jsonify({'status': 'error', 'message': 'Invalid URL format'})

    accessible, error = check_url_accessibility(url)
    if not accessible:
        return jsonify({'status': 'error', 'message': f'URL not accessible: {error}'})

    try:
        # Force smaller crawl limits for the web interface
        config.crawler.max_pages = 10
        config.crawler.max_depth = 2

        # Start crawling
        crawler = Crawler(config.crawler)
        pages = crawler.crawl(url)

        if not pages:
            return jsonify({'status': 'error', 'message': 'No pages found to index'})

        # Parse pages
        parser = Parser()
        parsed_contents = []
        for page_url, html_content in pages.items():
            try:
                parsed_content = parser.parse_page(page_url, html_content)
                parsed_contents.append(parsed_content)
            except Exception as e:
                continue

        # Generate embeddings
        indexer = Indexer(config.indexing)
        chunks = indexer.process_documents(parsed_contents)

        # Store in vector database
        storage = FAISSStorage(dimension=config.indexing.vector_dimension)
        storage.add_chunks(chunks)

        # Save the index
        domain = url.replace('http://', '').replace('https://', '').split('/')[0]
        index_dir = Path(config.storage.storage_dir) / f"index_{domain.replace('.', '_')}"
        storage.save(str(index_dir))

        # Set up query engine for this domain
        global current_domain, query_engine
        current_domain = domain
        query_engine = QueryEngine(
            vector_store=storage,
            top_k=config.query.top_k,
            confidence_threshold=config.query.confidence_threshold
        )

        return jsonify({
            'status': 'success',
            'message': f'Successfully indexed {len(chunks)} chunks from {len(parsed_contents)} pages',
            'domain': domain
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error during crawling: {str(e)}'})


@app.route('/query', methods=['POST'])
def answer_question():
    question = request.form.get('question')
    domain = request.form.get('domain')

    if not question:
        return jsonify({'status': 'error', 'message': 'Question is required'})

    global current_domain, query_engine

    # Check if we need to load a different domain
    if not query_engine or (domain and domain != current_domain):
        try:
            index_dir = Path(config.storage.storage_dir) / f"index_{domain.replace('.', '_')}"

            if not index_dir.exists():
                return jsonify({'status': 'error', 'message': f'No index found for {domain}. Crawl it first.'})

            storage = FAISSStorage(dimension=config.indexing.vector_dimension)
            storage.load(str(index_dir))

            query_engine = QueryEngine(
                vector_store=storage,
                top_k=config.query.top_k,
                confidence_threshold=config.query.confidence_threshold
            )
            current_domain = domain

        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error loading index: {str(e)}'})

    if not query_engine:
        return jsonify({'status': 'error', 'message': 'No index loaded. Crawl a website first.'})

    try:
        # Process the query
        result = query_engine.process_query(question)

        # Format the response
        formatter = ResponseFormatter(
            show_sources=True,
            show_confidence=True,
            max_sources=3
        )

        # Create a markdown response
        response_md = formatter.format_result(result, format_type="markdown")

        # Extract sources for better display
        sources = []
        for chunk, score in result.source_chunks[:3]:
            if score >= 0.6:  # Only include reasonably confident sources
                sources.append({
                    'url': chunk.metadata.get('url', ''),
                    'title': chunk.metadata.get('title', 'Unknown'),
                    'section': chunk.metadata.get('section_title', ''),
                    'confidence': f"{score:.2f}"
                })

        return jsonify({
            'status': 'success',
            'answer': result.answer,
            'confidence': f"{result.confidence:.2f}",
            'sources': sources,
            'markdown': response_md
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error processing query: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)