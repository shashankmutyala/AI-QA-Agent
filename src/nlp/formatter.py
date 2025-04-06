import logging
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .query_engine import QueryResult


class ResponseFormatter:
    """
    Formats query results for display to the user.
    Handles different output formats and includes source references.
    """

    def __init__(self,
                 show_sources: bool = True,
                 show_confidence: bool = True,
                 max_sources: int = 3,
                 min_confidence_for_source: float = 0.5):
        """
        Initialize the formatter.

        Args:
            show_sources: Whether to include source references
            show_confidence: Whether to show confidence scores
            max_sources: Maximum number of sources to include
            min_confidence_for_source: Minimum confidence score to include a source
        """
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        self.show_sources = show_sources
        self.show_confidence = show_confidence
        self.max_sources = max_sources
        self.min_confidence_for_source = min_confidence_for_source

    def _format_markdown(self, result: QueryResult) -> str:
        """
        Format the result as markdown text.

        Args:
            result: The query result to format

        Returns:
            Formatted markdown string
        """
        if not result.answer.strip():
            self.logger.warning("Empty answer provided. Formatting skipped.")
            return "**No valid answer was generated for the query.**"

        md_parts = [f"> {result.answer}"]  # Blockquote for the answer

        if self.show_sources and result.source_chunks:
            md_parts.append("\n\n**Sources:**")

            # Sort sources by relevance
            sorted_chunks = sorted(result.source_chunks, key=lambda x: x[1], reverse=True)

            for i, (chunk, score) in enumerate(sorted_chunks[:self.max_sources]):
                if score < self.min_confidence_for_source:
                    continue
                url = chunk.metadata.get('url', 'Unknown source')
                title = chunk.metadata.get('title', 'Untitled')
                section = chunk.metadata.get('section_title', '')
                source_line = f"{i + 1}. [{title}{' > ' + section if section else ''}]({url})"
                if self.show_confidence:
                    confidence_pct = int(score * 100)
                    source_line += f" (Relevance: {confidence_pct}%)"
                md_parts.append(source_line)

        if self.show_confidence:
            confidence_pct = int(result.confidence * 100)
            confidence_desc = (
                "High Confidence" if result.confidence >= 0.8 else
                "Medium Confidence" if result.confidence >= 0.5 else
                "Low Confidence"
            )
            md_parts.append(f"\n\n*Confidence: {confidence_desc} ({confidence_pct}%)*")

        return "\n".join(md_parts)

    def format_terminal(self, result: QueryResult) -> None:
        """
        Format and print the result to the terminal using rich formatting.

        Args:
            result: The query result to format
        """
        if not result.answer.strip():
            self.console.print("[bold red]No valid answer was generated for the query.[/]")
            return

        self.console.print(Panel(
            Markdown(result.answer),
            title="Query Answer",
            expand=False
        ))

        if self.show_confidence:
            confidence_pct = int(result.confidence * 100)
            confidence_color = (
                "green" if result.confidence >= 0.8 else
                "yellow" if result.confidence >= 0.5 else
                "red"
            )
            self.console.print(f"Confidence: [bold {confidence_color}]{confidence_pct}%[/]")

        if self.show_sources and result.source_chunks:
            table = Table(title="Sources")
            table.add_column("Source", style="cyan")
            table.add_column("Section", style="magenta")

            if self.show_confidence:
                table.add_column("Relevance", style="green")

            sorted_chunks = sorted(result.source_chunks, key=lambda x: x[1], reverse=True)

            for i, (chunk, score) in enumerate(sorted_chunks[:self.max_sources]):
                if score < self.min_confidence_for_source:
                    continue
                title = chunk.metadata.get('title', 'Untitled')
                section = chunk.metadata.get('section_title', '-')

                if self.show_confidence:
                    confidence_pct = int(score * 100)
                    table.add_row(title, section, f"{confidence_pct}%")
                else:
                    table.add_row(title, section)

            self.console.print(table)

    def format_result(self, result: QueryResult, format_type: str = "terminal") -> Optional[str]:
        """
        Format the query result based on the requested format type.

        Args:
            result: The query result to format
            format_type: The format type ("terminal", "markdown", or "plain")

        Returns:
            Formatted string for non-terminal outputs, None for terminal output
        """
        if format_type == "terminal":
            self.format_terminal(result)
            return None
        elif format_type == "markdown":
            return self._format_markdown(result)
        elif format_type == "plain":
            output = [result.answer]
            if self.show_sources and result.source_chunks:
                output.append("\nSources:")
                for i, (chunk, score) in enumerate(result.source_chunks[:self.max_sources]):
                    if score < self.min_confidence_for_source:
                        continue
                    url = chunk.metadata.get('url', 'Unknown source')
                    title = chunk.metadata.get('title', 'Untitled')
                    output.append(f"{i + 1}. {title} - {url}")
            return "\n".join(output)
        else:
            self.logger.warning(f"Unknown format type: {format_type}")
            return result.answer