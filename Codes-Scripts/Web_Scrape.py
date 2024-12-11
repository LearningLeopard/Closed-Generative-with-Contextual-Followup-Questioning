import json
import requests
from bs4 import BeautifulSoup


def get_tags_between_headings(start_tag, end_tag):
    tags = []
    current = start_tag.next_sibling
    while current and current != end_tag:
        tags.append(current)
        current = current.next_sibling
    return tags


# URL of the website to scrape
root_url = "https://www.immigrationhelp.org"
url = "https://www.immigrationhelp.org/browse"

# Send a GET request to the website
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    data = []
    # Find all h4 tags
    h4_tags = soup.find_all('h4')
    for h4 in h4_tags:

        h4_obj = {
            "title": h4.get_text(strip=True),
            "links": []
        }
        
        # Get the next sibling that is a tag (not just text)
        next_tag = h4.find_next_sibling()

        if next_tag:            
            # Find all child nodes of next_tag
            child_nodes = list(next_tag.children)
            for div_child in child_nodes:
                text = div_child.get_text(strip=True)
                a_tag = div_child.find('a')
                href = a_tag['href'] 
                link = root_url + href

                link_obj = {
                    "text": text,
                    "link": link,
                }
                qa_data = []
                link_response = requests.get(link)
                if link_response.status_code == 200:
                    link_soup = BeautifulSoup(link_response.content, 'html.parser')
                    article_body = link_soup.find('article', class_='content-body__container')

                    sections = [next(section.children) for section in article_body.find_all('section', recursive=False)]

                    for section in sections:
                        

                        content = []
                        headings = section.find_all(['h3', 'h4', 'h5', 'h6'])
                        # print(f"headings: {headings}")
                        
                        for i, heading in enumerate(headings):
                            qa_obj = {}
                            heading_text = heading.get_text(strip=True)
                            # print(f"heading_text: {heading_text}")
                            qa_obj["question"] = heading_text
                            # content.append(f"Heading: {heading_text}")
                            
                            current = heading.next_sibling
                            while current and (i == len(headings) - 1 or current != headings[i+1]):
                                content.append(current.get_text(strip=True))
                                current = current.next_sibling

                            qa_obj["answer"] = " ".join(content)
                            qa_data.append(qa_obj)

                link_obj["qa_data"] = qa_data
                print(f"link_obj: {link_obj} \n\n")

                h4_obj["links"].append(link_obj)

            data.append(h4_obj)

    with open('data.json', 'w') as f:
        json.dump(data, f)

else:
    print(f"Failed to retrieve content from {url}, status code: {response.status_code}")
