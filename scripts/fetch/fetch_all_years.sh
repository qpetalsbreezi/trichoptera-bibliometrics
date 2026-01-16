#!/bin/bash
# Fetch all years 2010-2025 from Scopus API

for year in {2010..2025}; do
    echo "=========================================="
    echo "Fetching year: $year"
    echo "=========================================="
    python3 scripts/fetch/fetch_scopus_api.py --year $year --view standard --output "data/raw/scopus_api/scopus_api_${year}.csv"
    echo ""
    sleep 1  # Brief pause between years
done

echo "=========================================="
echo "All years fetched!"
echo "=========================================="
