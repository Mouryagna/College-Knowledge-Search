import os
from pathlib import Path
from bs4 import BeautifulSoup
import pypdf
import docx
import pandas as pd


def extract_html_text(html_content, url):
    """Converts HTML text and tables into structured Markdown with metadata."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer"]):
        script.extract()

    title = soup.title.string.strip() if soup.title and soup.title.string else "IIIT Nagpur Document"

    # Convert HTML tables to Markdown tables
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            md_table = "\n" + "\n".join(rows) + "\n"
            table.replace_with(soup.new_string(md_table))

    text = soup.get_text(separator="\n", strip=True)
    header = f"Document Title: {title}\nSource URL: {url}\nCategory: Web Page\n---\n"
    return header + text


def extract_pdf_text(file_path, url):
    """Extracts text from PDF documents."""
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")

    filename = os.path.basename(file_path)
    header = f"Document Title: {filename}\nSource URL: {url}\nCategory: PDF Document\n---\n"
    return header + text


def extract_docx_text(file_path, url):
    """Extracts text from DOCX files."""
    text = ""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")

    filename = os.path.basename(file_path)
    header = f"Document Title: {filename}\nSource URL: {url}\nCategory: Word Document\n---\n"
    return header + text


def extract_excel_text(file_path, url):
    """Extracts text from Excel spreadsheets as Markdown tables."""
    text = ""
    try:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            text += f"\nSheet: {sheet_name}\n" + df.to_markdown(index=False) + "\n"
    except Exception as e:
        print(f"Error reading Excel {file_path}: {e}")

    filename = os.path.basename(file_path)
    header = f"Document Title: {filename}\nSource URL: {url}\nCategory: Spreadsheet\n---\n"
    return header + text


def process_item(item):
    """Routes crawled items to appropriate extractor."""
    if item["type"] == "html":
        return extract_html_text(item["content"], item["url"])
    elif item["type"] == "file":
        path = item["path"]
        url = item["url"]
        if path.endswith(".pdf"):
            return extract_pdf_text(path, url)
        elif path.endswith(".docx"):
            return extract_docx_text(path, url)
        elif path.endswith(".xlsx"):
            return extract_excel_text(path, url)
    return ""