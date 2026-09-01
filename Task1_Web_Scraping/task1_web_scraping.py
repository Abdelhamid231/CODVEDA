"""
Web Scraping - Quotes and Author Data
---------------------------------------
Scrapes quotes, authors, and tags from quotes.toscrape.com.
Handles pagination across multiple pages and collects author
bio information from individual profile pages.

Target: https://quotes.toscrape.com
Output: CSV and JSON files with scraped data
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import os
import re
from urllib.parse import urljoin



# CONFIGURATION

BASE_URL = "https://quotes.toscrape.com"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "scraped_quotes.csv")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "scraped_quotes.json")
AUTHOR_CSV_OUTPUT = os.path.join(OUTPUT_DIR, "scraped_authors.csv")
DELAY_BETWEEN_REQUESTS = 1  # seconds, to be respectful to the server



# SECTION 1: Website Inspection

def inspect_website(url):
    """
    Identify a target website and inspect its structure.
    
    This function fetches the homepage and inspects:
    - HTTP status code and headers
    - HTML structure (tags, classes, IDs)
    - Key elements for scraping (quotes, authors, tags, pagination)
    """
    print(" Website Inspection")

    print(f"\nTarget URL: {url}")
    
    # Send GET request
    response = requests.get(url)
    
    # HTTP Response
    print(f"\n--- HTTP Response Info ---")
    print(f"  Status Code : {response.status_code}")
    print(f"  Content-Type: {response.headers.get('Content-Type', 'N/A')}")
    print(f"  Encoding    : {response.encoding}")
    print(f"  Content Size: {len(response.content)} bytes")
    
    # HTML
    soup = BeautifulSoup(response.text, "lxml")
    
    # Page Structure
    print(f"\n--- Page Structure ---")
    print(f"  Title: {soup.title.string if soup.title else 'N/A'}")
    
    # Count key HTML elements
    all_tags = [tag.name for tag in soup.find_all(True)]
    unique_tags = set(all_tags)
    print(f"  Total HTML elements: {len(all_tags)}")
    print(f"  Unique tag types   : {len(unique_tags)}")
    print(f"  Tags found: {', '.join(sorted(unique_tags))}")
    
    # Scraping Targets
    print(f"\n--- Scraping Targets Identified ---")
    
    # Quotes
    quotes = soup.find_all("div", class_="quote")
    print(f"  Quotes on page      : {len(quotes)}")
    
    if quotes:
        first_quote = quotes[0]
        text = first_quote.find("span", class_="text")
        author = first_quote.find("small", class_="author")
        tags = first_quote.find_all("a", class_="tag")
        
        print(f"\n  Sample Quote Structure:")
        print(f"    Text element   : <span class='text'> -> {text.string[:50]}..." if text else "    Text: N/A")
        print(f"    Author element : <small class='author'> -> {author.string}" if author else "    Author: N/A")
        print(f"    Tags elements  : <a class='tag'> -> {[t.string for t in tags]}")
    
    # Pagination
    next_btn = soup.find("li", class_="next")
    print(f"\n  Pagination ('Next' button): {'Found' if next_btn else 'Not found'}")
    if next_btn:
        next_link = next_btn.find("a")
        print(f"    Next page URL: {next_link['href'] if next_link else 'N/A'}")
    
    print(f"\n{'=' * 70}")
    print("Website inspection complete. Structure is well-suited for scraping.")
    print(f"{'=' * 70}\n")
    
    return soup



# SECTION 2: Data Scraping with Pagination Handling

def scrape_single_page(url):
    """
    Scrape all quotes from a single page.
    
    Returns:
        list[dict]: List of quote dictionaries
        str or None: URL of the next page, or None if last page
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for HTTP errors
    
    soup = BeautifulSoup(response.text, "lxml")
    quotes_data = []
    
    quote_elements = soup.find_all("div", class_="quote")
    
    for quote_el in quote_elements:
        # Extract text (remove surrounding unicode quotes \u201c \u201d)
        text_raw = quote_el.find("span", class_="text").get_text()
        text_clean = text_raw.strip("\u201c\u201d").strip()
        
        # Extract author
        author = quote_el.find("small", class_="author").get_text().strip()
        
        # Extract author's detail page link
        author_link_el = quote_el.find("a", href=True)
        author_link = urljoin(BASE_URL, author_link_el["href"]) if author_link_el else None
        
        # Extract tags
        tag_elements = quote_el.find_all("a", class_="tag")
        tags = [tag.get_text().strip() for tag in tag_elements]
        
        quotes_data.append({
            "quote": text_clean,
            "author": author,
            "author_url": author_link,
            "tags": tags,
            "tags_str": ", ".join(tags),  # For CSV compatibility
            "source_url": url
        })
    
    # Pagination
    next_page_url = None
    next_btn = soup.find("li", class_="next")
    if next_btn:
        next_link = next_btn.find("a")
        if next_link:
            next_page_url = urljoin(BASE_URL, next_link["href"])
    
    return quotes_data, next_page_url


def scrape_all_quotes():
    """
    Use BeautifulSoup and requests to scrape data,
    handling pagination across all pages.
    
    Scrapes all quotes from https://quotes.toscrape.com by following
    the pagination links until no more pages are available.
    """
    print(" Scraping Quotes (with Pagination Handling)")

    all_quotes = []
    current_url = BASE_URL
    page_num = 1
    
    while current_url:
        print(f"\n  Scraping page {page_num}: {current_url}")
        
        try:
            quotes, next_url = scrape_single_page(current_url)
            all_quotes.extend(quotes)
            print(f"    -> Found {len(quotes)} quotes (Total so far: {len(all_quotes)})")
            
            current_url = next_url
            page_num += 1
            
            # Respectful delay between requests
            if current_url:
                time.sleep(DELAY_BETWEEN_REQUESTS)
                
        except requests.exceptions.RequestException as e:
            print(f"    -> ERROR: {e}")
            print("    -> Retrying in 3 seconds...")
            time.sleep(3)
            # Could implement retry logic here; for now, break
            break
    
    print(f"\n  Scraping complete!")
    print(f"  Total pages scraped : {page_num - 1}")
    print(f"  Total quotes scraped: {len(all_quotes)}")
    print(f"{'=' * 70}\n")
    
    return all_quotes


def scrape_author_details(author_urls):
    """
    Scrape detailed author information from author bio pages.
    
    
    Args:
        author_urls: dict mapping author name -> author detail URL
    
    Returns:
        list[dict]: Author detail records
    """
    print(" Scraping Author Details (Sub-page Navigation)")

    authors_data = []
    
    for i, (name, url) in enumerate(author_urls.items(), 1):
        print(f"\n  [{i}/{len(author_urls)}] Scraping author: {name}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract author details
            born_date = soup.find("span", class_="author-born-date")
            born_location = soup.find("span", class_="author-born-location")
            description = soup.find("div", class_="author-description")
            
            author_record = {
                "name": name,
                "born_date": born_date.get_text().strip() if born_date else "N/A",
                "born_location": born_location.get_text().strip() if born_location else "N/A",
                "bio": description.get_text().strip()[:300] + "..." if description else "N/A",
                "profile_url": url
            }
            
            authors_data.append(author_record)
            print(f"    Born: {author_record['born_date']} {author_record['born_location']}")
            
            time.sleep(DELAY_BETWEEN_REQUESTS)
            
        except requests.exceptions.RequestException as e:
            print(f"    -> ERROR scraping {name}: {e}")
            authors_data.append({
                "name": name,
                "born_date": "ERROR",
                "born_location": "ERROR",
                "bio": str(e),
                "profile_url": url
            })
    
    print(f"\n  Author scraping complete! Total authors: {len(authors_data)}")
    print(f"{'=' * 70}\n")
    
    return authors_data



# SECTION 3: Storing Data in Structured Formats (CSV & JSON)

def save_to_csv(data, filepath, columns=None):
    """
    Save data to CSV format using pandas.
    
    Args:
        data: list of dictionaries
        filepath: output CSV path
        columns: optional list of column names to include/order
    """
    df = pd.DataFrame(data)
    if columns:
        df = df[columns]
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"  CSV saved: {filepath}")
    print(f"    -> {len(df)} rows, {len(df.columns)} columns")
    print(f"    -> Columns: {list(df.columns)}")
    return df


def save_to_json(data, filepath):
    """
    Save data to JSON format with proper formatting.
    
    Args:
        data: list of dictionaries
        filepath: output JSON path
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(filepath)
    print(f"  JSON saved: {filepath}")
    print(f"    -> {len(data)} records")
    print(f"    -> File size: {file_size:,} bytes")


def store_data(quotes_data, authors_data):
    """
    Store scraped data in structured formats (CSV, JSON).
    """
    print(" Storing Data in Structured Formats")

    # Quotes
    print("\n--- Quotes Data ---")
    
    # CSV (using flat columns suitable for tabular format)
    quotes_csv_columns = ["quote", "author", "tags_str", "source_url"]
    quotes_df = save_to_csv(quotes_data, CSV_OUTPUT, columns=quotes_csv_columns)
    
    # JSON (preserving nested structure with tags as array)
    quotes_json_data = [
        {
            "quote": q["quote"],
            "author": q["author"],
            "author_url": q["author_url"],
            "tags": q["tags"],
            "source_url": q["source_url"]
        }
        for q in quotes_data
    ]
    save_to_json(quotes_json_data, JSON_OUTPUT)
    
    # Authors
    print("\n--- Authors Data ---")
    authors_df = save_to_csv(authors_data, AUTHOR_CSV_OUTPUT)
    
    print(f"\n{'=' * 70}")
    print("Data storage complete!")
    print(f"{'=' * 70}\n")
    
    return quotes_df, authors_df



# SECTION 4: Summary & Data Preview

def generate_summary(quotes_df, authors_df, quotes_data):
    """
    Generate a summary report of the scraped data.
    """
    print(" Scraping Summary Report")

    print(f"\n--- Dataset Overview ---")
    print(f"  Total quotes collected    : {len(quotes_df)}")
    print(f"  Total unique authors      : {quotes_df['author'].nunique()}")
    print(f"  Total pages scraped       : {quotes_df['source_url'].nunique()}")
    print(f"  Author profiles collected : {len(authors_df)}")
    
    # Top authors by quote count
    print(f"\n--- Top 5 Authors by Quote Count ---")
    top_authors = quotes_df["author"].value_counts().head(5)
    for rank, (author, count) in enumerate(top_authors.items(), 1):
        print(f"  {rank}. {author}: {count} quotes")
    
    # Tag analysis
    all_tags = []
    for q in quotes_data:
        all_tags.extend(q["tags"])
    tag_counts = pd.Series(all_tags).value_counts()
    
    print(f"\n--- Top 10 Tags ---")
    print(f"  Total unique tags: {len(tag_counts)}")
    for rank, (tag, count) in enumerate(tag_counts.head(10).items(), 1):
        print(f"  {rank}. #{tag}: {count} quotes")
    
    # Data preview
    print(f"\n--- Data Preview (First 5 Quotes) ---")
    preview = quotes_df.head(5)[["author", "quote"]].copy()
    preview["quote"] = preview["quote"].str[:60] + "..."
    print(preview.to_string(index=False))
    
    print(f"\n--- Output Files ---")
    print(f"  1. {CSV_OUTPUT}")
    print(f"  2. {JSON_OUTPUT}")
    print(f"  3. {AUTHOR_CSV_OUTPUT}")
    
    print(f"\n{'=' * 70}")
    print("Process Complete: Data Collection and Web Scraping")
    print(f"{'=' * 70}")



# MAIN EXECUTION

def main():
    """Main execution flow for Task 1."""

    print("  TASK 1: DATA COLLECTION AND WEB SCRAPING")

    print("=" * 70 + "\n")
    
    # Step 1: Inspect the website structure
    inspect_website(BASE_URL)
    
    # Step 2: Scrape all quotes with pagination handling
    all_quotes = scrape_all_quotes()
    
    if not all_quotes:
        print("ERROR: No quotes were scraped. Exiting.")
        return
    
    # Step 2b: Collect unique author URLs and scrape author details
    author_urls = {}
    for q in all_quotes:
        if q["author"] not in author_urls and q["author_url"]:
            author_urls[q["author"]] = q["author_url"]
    
    authors_data = scrape_author_details(author_urls)
    
    # Step 3: Store data in CSV and JSON formats
    quotes_df, authors_df = store_data(all_quotes, authors_data)
    
    # Step 4: Generate summary report
    generate_summary(quotes_df, authors_df, all_quotes)


if __name__ == "__main__":
    main()




