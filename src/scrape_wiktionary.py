import requests
from bs4 import BeautifulSoup
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
session = requests.Session()
session.headers.update(headers)

def parse_language_page(word_url, lang_name, lang_code, retries=5):
    response = None
    for attempt in range(retries):
        try:
            response = session.get(word_url, timeout=15)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get('Retry-After')
                wait = float(retry_after) if retry_after else 2 ** attempt
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue
                return None
            if response.status_code != 200:
                return None
            break
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    else:
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    lang_h2 = soup.find(lambda tag: tag.name == 'h2' and tag.get('id') == lang_name)
    if not lang_h2:
        return []
        
    parent_div = lang_h2.parent
    
    results = []
    current_ipa = ""
    current_ortho = ""
    
    current = parent_div.find_next_sibling()
    while current and not (current.name == 'div' and 'mw-heading2' in current.get('class', [])):
        if not hasattr(current, 'name') or not current.name:
            current = current.find_next_sibling()
            continue
            
        # check for IPA
        ipa_spans = current.find_all('span', class_='IPA')
        if ipa_spans:
            current_ipa = ipa_spans[0].text.strip()
            
        # check for headword
        headwords = current.find_all('strong', class_='headword', lang=lang_code)
        if headwords:
            current_ortho = headwords[0].text.strip()
            if current_ortho:
                results.append({
                    'orthography': current_ortho,
                    'ipa': current_ipa
                })
                current_ortho = "" # Reset for next POS if any
        
        current = current.find_next_sibling()
        
    return results

def get_category_links(lang_name):
    category_url = f'https://en.wiktionary.org/wiki/Category:{lang_name}_terms_with_IPA_pronunciation'
    links = []
    current_url = category_url
    
    while current_url:
        print(f"  Fetching category: {current_url}")
        response = requests.get(current_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        groups = soup.find_all('div', class_='mw-category-group')
        for group in groups:
            for a in group.find_all('a'):
                href = a.get('href')
                if href:
                    links.append('https://en.wiktionary.org' + href)
                    
        next_page = soup.find('a', string=re.compile('next page', re.I))
        if next_page and next_page.get('href'):
            current_url = 'https://en.wiktionary.org' + next_page.get('href')
        else:
            current_url = None
            
    return links

def fetch_all(links, lang_name, lang_code, max_workers):
    """Fetch every link, returning (results, failed_links). A link only
    ends up in failed_links if the request itself never succeeded (after
    retries) -- pages that loaded fine but had no matching section are
    counted as legitimate empty results, not failures."""
    results = []
    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(parse_language_page, link, lang_name, lang_code): link for link in links}
        for future in as_completed(futures):
            done += 1
            link = futures[future]
            res = future.result()
            if res is None:
                failed.append(link)
            else:
                results.extend(res)
            if done % 50 == 0 or done == len(links):
                print(f"  [{done}/{len(links)}] done...")
    return results, failed

def scrape_language(lang_name, lang_code, max_workers=10):
    print(f"Scraping {lang_name}...")
    links = get_category_links(lang_name)
    print(f"  Found {len(links)} links. Fetching with {max_workers} threads...")

    all_results, failed = fetch_all(links, lang_name, lang_code, max_workers)

    retry_round = 1
    while failed and retry_round <= 4:
        print(f"  {len(failed)} pages failed to fetch, retrying (round {retry_round})...")
        time.sleep(5 * retry_round)
        res, failed = fetch_all(failed, lang_name, lang_code, max(1, max_workers // 2))
        all_results.extend(res)
        retry_round += 1

    if failed:
        print(f"  WARNING: {len(failed)} pages could not be fetched after retries and were skipped:")
        for link in failed:
            print(f"    {link}")

    csv_file = f'{lang_name.lower()}_terms.csv'
    print(f"  Saving {len(all_results)} entries to {csv_file}")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['orthography', 'ipa'])
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
            
    print(f"  Done scraping {lang_name}!\n")

def main():
    languages = [
        {"name": "Karelian", "code": "krl"},
        {"name": "Livonian", "code": "liv"},
        {"name": "Ingrian", "code": "izh"}
    ]
    for lang in languages:
        scrape_language(lang["name"], lang["code"])

if __name__ == '__main__':
    main()

