import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add the project root to the Python path to fix imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.progress import Progress

from src.config import load_config, AppConfig
from src.crawling.crawler import Crawler
from src.crawling.parser import Parser, ParsedContent
from src.indexing.indexer import Indexer, ContentChunk
from src.indexing.storage import FAISSStorage
from src.nlp.query_engine import QueryEngine
from src.nlp.formatter import ResponseFormatter
from src.utils.validators import is_valid_url, validate_query, check_url_accessibility
from src.utils.error_handling import handle_request_error

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


def normalize_url(url: str) -> str:
    """Normalize the URL for consistency."""
    return url.strip().rstrip('/')


def crawl_and_index(url: str, config: AppConfig, reindex: bool = False) -> bool:
    """Crawl a help website and index its content."""
    storage_path = Path(config.storage.storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    domain = normalize_url(url).replace('http://', '').replace('https://', '').split('/')[0]
    index_dir = storage_path / f"index_{domain.replace('.', '_')}"

    if index_dir.exists() and not reindex:
        console.print(f"[yellow]Index for {domain} already exists. Use --reindex to rebuild it.[/]")
        return True

    crawler = Crawler(config.crawler)
    parser = Parser()
    indexer = Indexer(config.indexing)
    storage = FAISSStorage(dimension=config.indexing.vector_dimension)

    console.print(f"[bold blue]Crawling {url}...[/]")
    try:
        for attempt in range(3):  # Retry mechanism
            try:
                with Progress() as progress:
                    crawl_task = progress.add_task("[green]Crawling...", total=None)
                    pages = crawler.crawl(url)
                    progress.update(crawl_task, completed=True)
                break
            except Exception as e:
                logger.warning(f"Retrying crawling ({attempt + 1}/3): {str(e)}")
                if attempt == 2:
                    console.print(f"[bold red]Failed to crawl {url} after 3 attempts.[/]")
                    return False

        if not pages:
            console.print(f"[bold red]No pages found at {url}. Ensure the URL is correct or the site is accessible.[/]")
            return False

        console.print(f"[green]Successfully crawled {len(pages)} pages.[/]")

        console.print("[bold blue]Processing content...[/]")
        parsed_contents: List[ParsedContent] = []
        with Progress() as progress:
            parse_task = progress.add_task("[green]Parsing pages...", total=len(pages))

            for page_url, html_content in pages.items():
                try:
                    parsed_content = parser.parse_page(page_url, html_content)
                    parsed_contents.append(parsed_content)
                    progress.update(parse_task, advance=1)
                except Exception as e:
                    logger.error(f"Error parsing {page_url}: {str(e)}")
                    progress.update(parse_task, advance=1)

        console.print(f"[green]Successfully parsed {len(parsed_contents)} pages.[/]")

        console.print("[bold blue]Generating embeddings...[/]")
        chunks = indexer.process_documents(parsed_contents)
        console.print(f"[green]Generated {len(chunks)} content chunks.[/]")

        console.print("[bold blue]Adding to vector store...[/]")
        storage.add_chunks(chunks)

        console.print("[bold blue]Saving index to disk...[/]")
        storage.save(str(index_dir))

        console.print(f"[bold green]Successfully indexed {len(chunks)} chunks from {len(parsed_contents)} pages.[/]")
        console.print(f"[bold green]Index saved to {index_dir}[/]")

        return True

    except Exception as e:
        console.print(f"[bold red]Error during crawling and indexing: {str(e)}[/]")
        logger.error(f"Error during crawling and indexing: {str(e)}", exc_info=True)
        return False


def answer_query(query: str, url: str, config: AppConfig) -> bool:
    """Answer a question using the indexed content."""
    storage_path = Path(config.storage.storage_dir)
    domain = normalize_url(url).replace('http://', '').replace('https://', '').split('/')[0]
    index_dir = storage_path / f"index_{domain.replace('.', '_')}"

    if not index_dir.exists():
        console.print(f"[bold red]No index found for {domain}. Use the 'crawl' command to create one.[/]")
        return False

    try:
        valid, error_msg = validate_query(
            query,
            min_length=config.query.min_query_length,
            max_length=config.query.max_query_length,
            require_question_format=config.query.require_question_format
        )

        if not valid:
            console.print(f"[bold red]{error_msg}[/]")
            return False

        console.print("[bold blue]Loading index...[/]")
        storage = FAISSStorage(dimension=config.indexing.vector_dimension)
        storage.load(str(index_dir))

        query_engine = QueryEngine(
            vector_store=storage,
            top_k=config.query.top_k,
            confidence_threshold=config.query.confidence_threshold
        )

        console.print("[bold blue]Processing query...[/]")
        result = query_engine.process_query(query)

        formatter = ResponseFormatter(
            show_sources=config.output.show_sources,
            show_confidence=config.output.show_confidence,
            max_sources=config.output.max_sources,
            min_confidence_for_source=config.output.min_confidence_for_source
        )

        formatter.format_result(result, format_type=config.output.default_format)
        return True

    except Exception as e:
        console.print(f"[bold red]Error processing query: {str(e)}[/]")
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Help Website Q&A Agent")

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    crawl_parser = subparsers.add_parser('crawl', help='Crawl and index a help website')
    crawl_parser.add_argument('url', help='URL of the help website')
    crawl_parser.add_argument('--reindex', action='store_true', help='Force reindexing if index already exists')
    crawl_parser.add_argument('--config', help='Path to configuration file')

    query_parser = subparsers.add_parser('query', help='Answer a question using indexed content')
    query_parser.add_argument('url', help='URL of the help website')
    query_parser.add_argument('question', nargs='?', help='Question to answer')
    query_parser.add_argument('--interactive', '-i', action='store_true', help='Start interactive query mode')
    query_parser.add_argument('--config', help='Path to configuration file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        config_path = args.config if hasattr(args, 'config') and args.config else "config/default_config.json"
        config = load_config(config_path)
    except Exception as e:
        console.print(
            f"[bold yellow]Warning: Couldn't load configuration from {config_path}, using defaults. Error: {str(e)}[/]")
        config = load_config(None)  # Use defaults

    if args.command == 'crawl':
        if not is_valid_url(args.url):
            console.print(f"[bold red]Invalid URL: {args.url}[/]")
            return

        accessible, error = check_url_accessibility(args.url)
        if not accessible:
            console.print(f"[bold red]URL not accessible: {error}[/]")
            return

        crawl_and_index(args.url, config, args.reindex)

    elif args.command == 'query':
        if not is_valid_url(args.url):
            console.print(f"[bold red]Invalid URL: {args.url}[/]")
            return

        if args.interactive:
            console.print("[bold green]Interactive query mode. Type 'exit' to quit.[/]")
            while True:
                query = console.input("[bold blue]Ask a question: [/]").strip()
                if query.lower() in ('exit', 'quit', 'q'):
                    console.print("[bold yellow]Exiting interactive mode.[/]")
                    break

                valid, error_msg = validate_query(
                    query,
                    min_length=config.query.min_query_length,
                    max_length=config.query.max_query_length,
                    require_question_format=config.query.require_question_format
                )
                if not valid:
                    console.print(f"[bold red]{error_msg}[/]")
                    continue

                answer_query(query, args.url, config)
                console.print()
        elif args.question:
            answer_query(args.question, args.url, config)
        else:
            console.print("[bold red]No question provided. Use --interactive for interactive mode.[/]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/]")
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        sys.exit(1)