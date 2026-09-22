import urllib.request
import urllib.robotparser
from pathlib import Path

BASE_URL = "https://www.fortunecloudindia.com"
OUTPUT_FILE = Path("data/raw/fortune_cloud_llms_full.md")

def get_llms_txt_url():
    """
    Parses robots.txt to discover the llms-full.txt location.
    Falls back to the standard /llms-full.txt if not found.
    """
    print(f"Checking {BASE_URL}/robots.txt...")
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/robots.txt", 
            headers={'User-Agent': 'Mozilla/5.0 (FortuneCloudBot/1.0)'}
        )
        response = urllib.request.urlopen(req)
        lines = response.read().decode('utf-8').splitlines()
        
        for line in lines:
            # Check for standard LLMs.txt directive
            if "LLMs-full.txt:" in line:
                url = line.split("LLMs-full.txt:")[1].strip()
                return url
            elif "LLMs.txt:" in line:
                url = line.split("LLMs.txt:")[1].strip()
                return url
                
    except Exception as e:
        print(f"Error fetching robots.txt: {e}")
        
    # Default fallback according to the llmstxt.org standard
    return f"{BASE_URL}/llms-full.txt"

def main():
    print("=" * 60)
    print("FORTUNE CLOUD — FAST INGESTION CRAWLER")
    print("=" * 60)
    
    llms_url = get_llms_txt_url()
    print(f"Discovered Markdown URL : {llms_url}")
    
    # Fetch content
    try:
        print("Downloading             : Please wait...")
        req = urllib.request.Request(
            llms_url, 
            headers={'User-Agent': 'Mozilla/5.0 (FortuneCloudBot/1.0)'}
        )
        response = urllib.request.urlopen(req)
        content = response.read().decode('utf-8')
        
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"Status                  : SUCCESS")
        print(f"Downloaded              : {len(content):,} characters")
        print(f"Saved to                : {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Status                  : FAILED")
        print(f"Error                   : {e}")
        
    print("=" * 60)

if __name__ == "__main__":
    main()