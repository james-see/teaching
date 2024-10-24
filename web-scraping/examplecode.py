import requests
from bs4 import BeautifulSoup

# Send a request to your blog
response = requests.get('https://sometimeslesswrong.com/blog/')

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Extract and print article titles
titles = soup.find_all('h4', class_='title')
for title in titles:
    print(title.text.strip())

