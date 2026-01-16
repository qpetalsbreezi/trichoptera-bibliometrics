"""
Fetch Trichoptera papers from Scopus API directly.
This bypasses the 200-result limit from Publish or Perish by using date ranges and pagination.

SETUP:
1. Get a Scopus API key from: https://dev.elsevier.com/
2. Add to .env file: SCOPUS_API_KEY=your_api_key_here
3. (Optional) If your institution requires it, add: SCOPUS_INST_TOKEN=your_inst_token_here

USAGE EXAMPLES:
    # Fetch all papers from 2023 (single query)
    python fetch_scopus_api.py --year 2023
    
    # Fetch papers from a specific date range
    python fetch_scopus_api.py --start-date 2023-01-01 --end-date 2023-12-31
    
    # Fetch 2023 by QUARTERS and auto-merge (recommended for large datasets)
    python fetch_scopus_api.py --start-date 2023-01-01 --end-date 2023-12-31 --period quarterly --merge
    
    # Fetch 2023 by MONTHS and auto-merge
    python fetch_scopus_api.py --start-date 2023-01-01 --end-date 2023-12-31 --period monthly --merge
    
    # Fetch quarterly WITHOUT merging (keeps separate files)
    python fetch_scopus_api.py --start-date 2023-01-01 --end-date 2023-12-31 --period quarterly
    
    # Fetch with complete view (includes abstracts, slower but more data)
    python fetch_scopus_api.py --year 2023 --view complete
    
    # Limit number of results (for testing)
    python fetch_scopus_api.py --year 2023 --max-results 50
    
    # Specify custom output file (for merged results)
    python fetch_scopus_api.py --start-date 2023-01-01 --end-date 2023-12-31 --period quarterly --merge --output data/raw/scopus_api/scopus_2023_complete.csv

NOTES:
- Default count is 25 results per page (most Scopus API tiers have this limit)
- The "standard" view provides basic metadata (faster) - DOES NOT include abstracts
- The "complete" view includes abstracts but requires premium API access (401 error if not available)
- IMPORTANT LIMITATIONS:
  1. Standard view only returns the FIRST author in dc:creator field
     (Same limitation as Publish or Perish)
  2. Standard view does NOT include abstracts (dc:description is empty)
- RECOMMENDED WORKFLOW:
  1. Fetch papers with this script (gets metadata + first author)
  2. Run scripts/fetch/fetch_abstracts.py to add abstracts (uses OpenAlex/Semantic Scholar)
  3. Run scripts/fetch/fetch_authors.py to add full author lists (uses OpenAlex)
- The script handles pagination automatically using cursor-based navigation
- Rate limiting: 0.5 second delay between requests (adjust if needed)
"""

import pandas as pd
import requests
import time
import argparse
from datetime import datetime
from tqdm import tqdm
import os
import json
from pathlib import Path

# Get project root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables from .env file if it exists
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

# Configuration
SCOPUS_API_KEY = os.getenv("SCOPUS_API_KEY")
SCOPUS_INST_TOKEN = os.getenv("SCOPUS_INST_TOKEN")  # Optional, may be required by some subscriptions

if not SCOPUS_API_KEY:
    raise ValueError("SCOPUS_API_KEY not set. Please set it in .env file or environment variable.")

BASE_URL = "https://api.elsevier.com/content/search/scopus"

# Query for Trichoptera papers
# Using TITLE-ABS-KEY to search in title, abstract, and keywords
TRICHOPTERA_QUERY = 'TITLE-ABS-KEY("Trichoptera" OR "caddisfly" OR "caddisflies" OR "caddis fly" OR "caddis flies")'

def fetch_scopus_papers(query, start_date=None, end_date=None, year=None, view="standard", max_results=None):
    """
    Fetch papers from Scopus API with date filtering and pagination.
    
    Args:
        query: Scopus query string
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        year: Year as integer (alternative to start_date/end_date)
        view: API view type ("standard" or "complete")
        max_results: Maximum number of results to fetch (None for all)
    
    Returns:
        List of paper records
    """
    all_results = []
    cursor = None  # Will be set after first request
    # Most Scopus API subscriptions limit count to 25 per request
    count = 25  # Safe default for all views
    start = 0  # Start position for first request
    
    # Build date filter
    # Note: Scopus API supports PUBYEAR for year-level queries
    # For date ranges, we use PUBYEAR with optional PUBMONTH
    if year:
        date_filter = f"PUBYEAR = {year}"
    elif start_date and end_date:
        # Convert dates to Scopus date format
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_year = start_dt.year
        start_month = start_dt.month
        end_year = end_dt.year
        end_month = end_dt.month
        
        if start_year == end_year and start_month == end_month:
            # Single month - use PUBYEAR and PUBMONTH
            date_filter = f"PUBYEAR = {start_year} AND PUBMONTH = {start_month}"
        elif start_year == end_year:
            # Same year, different months
            date_filter = f"PUBYEAR = {start_year} AND PUBMONTH >= {start_month} AND PUBMONTH <= {end_month}"
        else:
            # Different years - use year range
            # Note: For cross-year ranges, we'll fetch all years and filter in post-processing if needed
            # For now, use PUBYEAR range
            date_filter = f"PUBYEAR >= {start_year} AND PUBYEAR <= {end_year}"
    else:
        date_filter = None
    
    # Combine query with date filter
    if date_filter:
        full_query = f"{query} AND ({date_filter})"
    else:
        full_query = query
    
    print(f"Query: {full_query}")
    print(f"View: {view}, Count per page: {count}")
    
    headers = {
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "Accept": "application/json"
    }
    
    page = 0
    total_found = None
    
    while True:
        params = {
            "query": full_query,
            "count": count,
            "view": view,
            "httpAccept": "application/json",
            "sort": "pubdate"  # Sort by publication date
        }
        
        # Use cursor for pagination if available, otherwise use start
        if cursor:
            params["cursor"] = cursor
        else:
            params["start"] = start
        
        if SCOPUS_INST_TOKEN:
            params["insttoken"] = SCOPUS_INST_TOKEN
        
        try:
            response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
            
            # Better error handling - print response details
            if response.status_code != 200:
                print(f"\nAPI Error Details:")
                print(f"Status Code: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error Response: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Error Response Text: {response.text[:500]}")
            
            response.raise_for_status()
            
            data = response.json()
            search_results = data.get("search-results", {})
            
            # Get total count on first page
            if total_found is None:
                total_str = search_results.get("@totalResults", "0")
                try:
                    total_found = int(total_str)
                except (ValueError, TypeError):
                    total_found = 0
                print(f"Total results found: {total_found}")
                if max_results:
                    total_found = min(total_found, max_results)
            
            entries = search_results.get("entry", [])
            if not entries:
                break
            
            all_results.extend(entries)
            page += 1
            
            print(f"Page {page}: Fetched {len(entries)} records (Total: {len(all_results)})")
            
            # Check if we've reached max_results
            if max_results and len(all_results) >= max_results:
                all_results = all_results[:max_results]
                break
            
            # Get next cursor for pagination
            cursor_info = search_results.get("cursor", {})
            next_cursor = cursor_info.get("@next")
            
            if not next_cursor:
                # Try alternative: check if there are more results using start
                current_results = len(entries)
                if current_results < count:
                    break  # Last page (got fewer results than requested)
                
                # Try next page using start parameter
                # Only stop if we got fewer results than requested (last page)
                # Don't rely on total_found if it's 0 (API limitation)
                if max_results and len(all_results) >= max_results:
                    break
                
                # Continue to next page
                start += count
                cursor = None  # Reset cursor since we're using start
                
                # Only stop if total_found is valid and we've exceeded it
                if total_found and total_found > 0 and start >= total_found:
                    break
            else:
                cursor = next_cursor
                start = None  # Using cursor, not start
            
            # Rate limiting: be respectful to the API
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            if hasattr(e.response, 'status_code'):
                print(f"Status code: {e.response.status_code}")
                if e.response.status_code == 429:
                    print("Rate limit exceeded. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                elif e.response.status_code == 401:
                    print("Authentication error. Check your API key.")
                    break
            raise
    
    print(f"\nTotal records fetched: {len(all_results)}")
    return all_results


def parse_scopus_entry(entry):
    """
    Parse a Scopus API entry into a dictionary matching the existing CSV format.
    """
    # Extract basic fields
    scopus_id = entry.get("dc:identifier", "").replace("SCOPUS_ID:", "")
    title = entry.get("dc:title", "")
    eid = entry.get("eid", "")
    
    # Authors
    # Scopus API returns authors in dc:creator field
    # Standard view: returns first author as a string (e.g., "Smith J.")
    # Complete view: returns author array (not available in most API tiers)
    creator = entry.get("dc:creator", "")
    author_names = []  # Initialize list
    
    if isinstance(creator, list):
        # If it's a list (complete view), parse each author
        for author in creator:
            if isinstance(author, dict):
                given_name = author.get("given-name", "")
                surname = author.get("surname", "")
                if given_name and surname:
                    author_names.append(f"{given_name} {surname}")
                elif surname:
                    author_names.append(surname)
            elif isinstance(author, str):
                author_names.append(author)
    elif isinstance(creator, str):
        # Standard view: single string, may be semicolon-separated
        if ";" in creator:
            # Multiple authors separated by semicolon
            author_names = [a.strip() for a in creator.split(";")]
        else:
            # Single author
            if creator.strip():
                author_names = [creator.strip()]
    
    authors_str = ", ".join(author_names) if author_names else ""
    
    # Publication info
    pub_name = entry.get("prism:publicationName", "")
    publisher = entry.get("prism:publisher", "")
    year = entry.get("prism:coverDate", "")
    if year:
        try:
            year = int(year.split("-")[0])  # Extract year from YYYY-MM-DD
        except:
            year = None
    else:
        year = None
    
    # DOI
    doi = entry.get("prism:doi", "")
    
    # Citation count
    citation_count = entry.get("citedby-count", "")
    try:
        citation_count = int(citation_count) if citation_count else 0
    except:
        citation_count = 0
    
    # Abstract
    abstract = entry.get("dc:description", "")
    
    # Document type
    subtype = entry.get("subtypeDescription", "")
    subtype_description = entry.get("subtype", "")
    
    # Volume, Issue, Pages
    volume = entry.get("prism:volume", "")
    issue = entry.get("prism:issueIdentifier", "")
    page_range = entry.get("prism:pageRange", "")
    start_page = entry.get("prism:startingPage", "")
    end_page = entry.get("prism:endingPage", "")
    
    # ISSN
    issn = entry.get("prism:issn", "")
    
    # URLs
    link = entry.get("link", [])
    if isinstance(link, dict):
        link = [link]
    
    article_url = ""
    citation_url = ""
    for l in link:
        if l.get("@ref") == "scopus":
            article_url = l.get("@href", "")
        elif l.get("@ref") == "scopus-citedby":
            citation_url = l.get("@href", "")
    
    # Build record matching existing CSV structure
    record = {
        "Cites": citation_count,
        "Authors": authors_str,
        "Title": title,
        "Year": year,
        "Source": pub_name,
        "Publisher": publisher,
        "ArticleURL": article_url,
        "CitesURL": citation_url,
        "GSRank": "",  # Not available from Scopus API
        "QueryDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Type": subtype or subtype_description or "Article",
        "DOI": doi,
        "ISSN": issn,
        "CitationURL": f"https://api.elsevier.com/content/abstract/scopus_id/{scopus_id}" if scopus_id else "",
        "Volume": volume,
        "Issue": issue,
        "StartPage": start_page,
        "EndPage": end_page,
        "ECC": "",  # Not available from Scopus API
        "CitesPerYear": round(citation_count / (2025 - year), 2) if year and year < 2025 else citation_count,
        "CitesPerAuthor": round(citation_count / len(author_names), 2) if author_names else citation_count,
        "AuthorCount": len(author_names),
        "Age": 2025 - year if year else "",
        "Abstract": abstract,
        "FullTextURL": "",
        "RelatedURL": "",
        "ScopusID": scopus_id,
        "EID": eid
    }
    
    return record


def merge_and_deduplicate(files, output_file):
    """Merge multiple CSV files and remove duplicates"""
    all_dataframes = []
    
    for file in files:
        if Path(file).exists():
            try:
                df = pd.read_csv(file)
                all_dataframes.append(df)
                print(f"  Loaded {Path(file).name}: {len(df)} papers")
            except Exception as e:
                print(f"  Error loading {file}: {e}")
                continue
    
    if not all_dataframes:
        print("No files to merge!")
        return None
    
    # Combine all dataframes
    print(f"\nMerging {len(all_dataframes)} files...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    initial_count = len(combined_df)
    print(f"Total papers before deduplication: {initial_count}")
    
    # Remove duplicates by DOI
    if 'DOI' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['DOI'], keep='first')
        doi_deduped = initial_count - len(combined_df)
        if doi_deduped > 0:
            print(f"  Removed {doi_deduped} duplicates by DOI")
    
    # Remove duplicates by title (for papers without DOI)
    if 'Title' in combined_df.columns:
        before_title_dedup = len(combined_df)
        no_doi = combined_df['DOI'].isna() | (combined_df['DOI'] == '')
        if no_doi.sum() > 0:
            combined_df['Title_Normalized'] = combined_df['Title'].fillna('').str.lower().str.strip()
            mask = ~combined_df.duplicated(subset=['Title_Normalized'], keep='first')
            combined_df = combined_df[mask]
            title_deduped = before_title_dedup - len(combined_df)
            if title_deduped > 0:
                print(f"  Removed {title_deduped} duplicates by Title")
            combined_df = combined_df.drop(columns=['Title_Normalized'])
    
    print(f"Total papers after deduplication: {len(combined_df)}")
    
    # Save merged file
    combined_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(combined_df)} unique papers to {output_file}")
    
    return combined_df


def get_periods(start_date, end_date, period):
    """Generate date periods (monthly, quarterly, or yearly)"""
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        raise ImportError("python-dateutil package required. Install with: pip install python-dateutil")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    periods = []
    
    current = start
    while current <= end:
        if period == "monthly":
            period_end = current + relativedelta(months=1, days=-1)
            if period_end > end:
                period_end = end
            periods.append((
                current.strftime("%Y-%m-%d"),
                period_end.strftime("%Y-%m-%d")
            ))
            current = period_end + relativedelta(days=1)
        elif period == "quarterly":
            period_end = current + relativedelta(months=3, days=-1)
            if period_end > end:
                period_end = end
            periods.append((
                current.strftime("%Y-%m-%d"),
                period_end.strftime("%Y-%m-%d")
            ))
            current = period_end + relativedelta(days=1)
        elif period == "yearly":
            period_end = current + relativedelta(years=1, days=-1)
            if period_end > end:
                period_end = end
            periods.append((
                current.strftime("%Y-%m-%d"),
                period_end.strftime("%Y-%m-%d")
            ))
            current = period_end + relativedelta(days=1)
    
    return periods


def main():
    parser = argparse.ArgumentParser(description="Fetch Trichoptera papers from Scopus API")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--year", type=int, help="Year (alternative to start-date/end-date)")
    parser.add_argument("--period", type=str, choices=["monthly", "quarterly", "yearly"],
                       help="Split date range into periods and fetch separately (requires --start-date and --end-date)")
    parser.add_argument("--merge", action="store_true",
                       help="Automatically merge period results into single file (use with --period)")
    parser.add_argument("--view", type=str, default="standard", choices=["standard", "complete"],
                       help="API view type (standard=basic metadata, complete=full including abstract)")
    parser.add_argument("--max-results", type=int, help="Maximum number of results to fetch")
    parser.add_argument("--output", type=str, help="Output CSV file path")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.year and (args.start_date or args.end_date):
        print("Error: Cannot specify both --year and --start-date/--end-date")
        return
    
    if args.period and not (args.start_date and args.end_date):
        print("Error: --period requires both --start-date and --end-date")
        return
    
    if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
        print("Error: Both --start-date and --end-date must be provided together")
        return
    
    if not args.year and not args.start_date:
        print("Error: Must specify either --year or --start-date/--end-date")
        return
    
    # Handle period-based fetching
    if args.period:
        if not args.start_date or not args.end_date:
            print("Error: --period requires --start-date and --end-date")
            return
        
        try:
            from dateutil.relativedelta import relativedelta
        except ImportError:
            print("Error: python-dateutil package required for --period option")
            print("Install with: pip install python-dateutil")
            return
        
        periods = get_periods(args.start_date, args.end_date, args.period)
        print(f"\nSplitting into {len(periods)} {args.period} periods:")
        for i, (p_start, p_end) in enumerate(periods, 1):
            print(f"  {i}. {p_start} to {p_end}")
        
        period_files = []
        for i, (p_start, p_end) in enumerate(periods, 1):
            print(f"\n{'='*60}")
            print(f"Fetching period {i}/{len(periods)}: {p_start} to {p_end}")
            print(f"{'='*60}")
            
            period_start_str = p_start.replace("-", "")
            period_end_str = p_end.replace("-", "")
            period_output = PROJECT_ROOT / f"data/raw/scopus_api/scopus_api_{period_start_str}_{period_end_str}.csv"
            period_output.parent.mkdir(parents=True, exist_ok=True)
            
            results = fetch_scopus_papers(
                query=TRICHOPTERA_QUERY,
                start_date=p_start,
                end_date=p_end,
                year=None,
                view=args.view,
                max_results=args.max_results
            )
            
            if results:
                records = []
                for entry in tqdm(results, desc="Parsing entries"):
                    try:
                        record = parse_scopus_entry(entry)
                        records.append(record)
                    except Exception as e:
                        print(f"Error parsing entry: {e}")
                        continue
                
                df = pd.DataFrame(records)
                df.to_csv(period_output, index=False)
                print(f"✓ Saved {len(df)} records to {period_output.name}")
                period_files.append(period_output)
            else:
                print(f"No results for period {p_start} to {p_end}")
        
        # Merge if requested
        if args.merge and period_files:
            print(f"\n{'='*60}")
            print("Merging period results...")
            print(f"{'='*60}")
            
            if args.output:
                merged_output = Path(args.output)
                if not merged_output.is_absolute():
                    merged_output = PROJECT_ROOT / merged_output
            else:
                merged_output = PROJECT_ROOT / f"data/raw/scopus_api/scopus_api_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}_merged.csv"
            
            merged_df = merge_and_deduplicate(period_files, merged_output)
            
            if merged_df is not None:
                print("\n=== Final Summary ===")
                print(f"Total unique papers: {len(merged_df)}")
                if "Year" in merged_df.columns:
                    print(f"Year range: {merged_df['Year'].min()} - {merged_df['Year'].max()}")
                if "Source" in merged_df.columns:
                    print(f"Unique journals: {merged_df['Source'].nunique()}")
        
        return
    
    # Determine output filename
    if args.output:
        output_file = Path(args.output)
        if not output_file.is_absolute():
            output_file = PROJECT_ROOT / output_file
    elif args.year:
        output_file = PROJECT_ROOT / f"data/raw/scopus_api/scopus_api_{args.year}.csv"
    elif args.start_date and args.end_date:
        start_str = args.start_date.replace("-", "")
        end_str = args.end_date.replace("-", "")
        output_file = PROJECT_ROOT / f"data/raw/scopus_api/scopus_api_{start_str}_{end_str}.csv"
    else:
        output_file = PROJECT_ROOT / "data/raw/scopus_api/scopus_api_results.csv"
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Fetch papers
    print("Fetching papers from Scopus API...")
    print(f"Query: {TRICHOPTERA_QUERY}")
    
    results = fetch_scopus_papers(
        query=TRICHOPTERA_QUERY,
        start_date=args.start_date,
        end_date=args.end_date,
        year=args.year,
        view=args.view,
        max_results=args.max_results
    )
    
    if not results:
        print("No results found.")
        return
    
    # Parse results
    print("\nParsing results...")
    records = []
    for entry in tqdm(results, desc="Parsing entries"):
        try:
            record = parse_scopus_entry(entry)
            records.append(record)
        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue
    
    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} records to {output_file}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total papers: {len(df)}")
    if "Year" in df.columns:
        print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    if "Source" in df.columns:
        print(f"Unique journals: {df['Source'].nunique()}")
    if "Abstract" in df.columns:
        abstracts_count = df['Abstract'].notna().sum()
        print(f"Papers with abstracts: {abstracts_count} ({abstracts_count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
