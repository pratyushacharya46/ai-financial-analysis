from pathlib import Path
from edgar import *

def fetch_tesla_10k():
    set_identity("zeusfill03@gmail.com")
    
    data_dir = Path("core/data")
    output_file = data_dir / "tsla_10k.txt"
    
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print("Connecting to SEC EDGAR database for TSLA...")
        tesla = Company("TSLA")
        
        print("Locating the latest 10-K filing...")
        filings = tesla.get_filings(form="10-K")
        
        if not filings:
            print("10-K filings not found")
            return
            
        # Select the most recent one (index 0)
        latest_10k = filings[0]

        print("Extracting text")
        filing_text = latest_10k.text()
        
        print(f"Saving extracted text to {output_file}")
        # Specify utf-8 encoding to prevent Windows character mapping errors
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(filing_text)
            
        print(f"\nSaved {len(filing_text):,} characters to {output_file}")
        
    except Exception as e:
        print("\nAn error occurred during the data extraction pipeline:")
        print(f"Error Details: {e}")

if __name__ == "__main__":
    fetch_tesla_10k()