# src/ingestion/ingest_pipeline.py
import json
from pathlib import Path
import click
from .pdf_loader import extract_text_by_page, guess_chapters_from_headings
from .text_cleaner import clean_text
from .tokenizer import Tokenizer
from .splitter import chunk_pages

@click.command()
@click.option("--pdf", "pdf_path", required=True, help="Path to editable PDF")
@click.option("--outdir", default="data", help="Output directory for chunks (data/<book_slug>/chunks.jsonl)")
@click.option("--max-tokens", default=800, help="Max tokens per chunk")
@click.option("--overlap-tokens", default=128, help="Overlap tokens between chunks")
@click.option("--slug", default=None, help="Optional book slug for output folder")
def ingest(pdf_path: str, outdir: str, max_tokens: int, overlap_tokens: int, slug: str):
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    pages = extract_text_by_page(str(pdf_path))
    if not pages:
        raise SystemExit("No pages extracted from PDF — ensure the PDF is an editable/text PDF.")
    pages_dicts = []
    for p in pages:
        cleaned = clean_text(p.text)
        meta = p.metadata or {}
        pages_dicts.append({
            "page_number": p.page_number,
            "text": cleaned,
            "metadata": meta
        })

    chapters = guess_chapters_from_headings(pages)
    book_title = (pages[0].metadata.get("title") or pdf_path.stem) if pages else pdf_path.stem
    # safe slug: lowercase, replace spaces with underscore
    book_slug = slug or book_title.lower().replace(" ", "_")
    out_path = Path(outdir) / book_slug
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer()
    chunks = list(chunk_pages(pages_dicts, tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens))

    output_file = out_path / "chunks.jsonl"
    with open(output_file, "w", encoding="utf-8") as fh:
        for i, c in enumerate(chunks, start=1):
            doc = {
                "id": c["id"],
                "book_title": book_title,
                "book_slug": book_slug,
                "chunk_index": i,
                "text": c["text"],
                "token_count": c["token_count"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "source": f"{pdf_path.name}#pages={c['page_start']}-{c['page_end']}",
                "metadata": c.get("metadata", {}),
            }
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    click.echo(f"Wrote {len(chunks)} chunks to {output_file}")
    if chapters:
        with open(out_path / "chapters.json", "w", encoding="utf-8") as fh:
            fh.write(json.dumps(chapters, ensure_ascii=False, indent=2))
        click.echo(f"Wrote {len(chapters)} inferred chapters to {out_path / 'chapters.json'}")

if __name__ == "__main__":
    ingest()
