"""Extract title, footnotes, and table grid from a PNG via PaddleOCR-VL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from paddleocr import PaddleOCRVL

from chemtables.workers.paddleocr_vl.title_merge import merge_titles_above_table


def html_table_to_grid(html: str) -> list[list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No <table> found in HTML content.")
    rows = table.find_all("tr")

    grid = []
    rowspan_tracker = {}

    for r_idx, tr in enumerate(rows):
        row = []
        c_idx = 0
        cells = tr.find_all(["td", "th"])
        cell_iter = iter(cells)

        while True:
            while (r_idx, c_idx) in rowspan_tracker:
                row.append(rowspan_tracker[(r_idx, c_idx)])
                c_idx += 1

            try:
                cell = next(cell_iter)
            except StopIteration:
                break

            text = cell.get_text(strip=True)
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            for i in range(colspan):
                row.append(text)
                if rowspan > 1:
                    for r_offset in range(1, rowspan):
                        rowspan_tracker[(r_idx + r_offset, c_idx + i)] = text

            c_idx += colspan

        grid.append(row)

    return grid


def extract_table(input_path: Path, output_dir: Path) -> dict:
    """Run PaddleOCR-VL and write handoff JSON under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paddleocr_dir = output_dir / "paddleocr"
    paddleocr_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PaddleOCRVL(
        device="gpu:0",
        use_layout_detection=True,
        precision="fp16",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        layout_detection_model_name="PP-DocLayoutV2"
    )

    pages = list(pipeline.predict(input=str(input_path)))
    if not pages:
        print("No results returned.", file=sys.stderr)
        sys.exit(2)

    # First page/item with a table block wins (typical single-table crops).
    title = None
    footnotes = None
    grid = None

    for i, res in enumerate(pages, start=1):
        print(f"Processed page/item {i}")
        res.save_to_json(save_path=paddleocr_dir)
        res.save_to_markdown(save_path=paddleocr_dir)

        blocks = res.json["res"]["parsing_res_list"]
        table_block = next((b for b in blocks if b["block_label"] == "table"), None)
        if table_block is None:
            continue

        title = merge_titles_above_table(blocks)
        footnote_blocks = [
            b for b in blocks if b["block_label"] in {"vision_footnote", "footnote"}
        ]
        footnotes = (
            "\n".join(b["block_content"] for b in footnote_blocks)
            if footnote_blocks
            else None
        )
        grid = html_table_to_grid(table_block["block_content"])
        break

    if grid is None:
        print("No table block found in OCR results.", file=sys.stderr)
        sys.exit(3)

    payload = {
        "source_image": str(input_path.resolve()),
        "title": title,
        "footnotes": footnotes,
        "grid": grid,
    }
    handoff_path = output_dir / "extracted_table.json"
    with open(handoff_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Wrote handoff: {handoff_path.resolve()}")
    return payload

if __name__ == "__main__":
    input_path = Path("images")
    output_dir = Path("output")
    extract_table(input_path, output_dir)
