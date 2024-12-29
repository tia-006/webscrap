import requests
from bs4 import BeautifulSoup
import csv

url = "http://books.toscrape.com/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    products = soup.find_all('article', class_='product_pod')

    data = []

    for product in products:
        title_element = product.find('h3').find('a')
        price_element = product.find('p', class_='price_color')

        if title_element and price_element:
            title = title_element['title'] if 'title' in title_element.attrs else title_element.get_text(strip=True)
            price = price_element.get_text(strip=True)
            data.append([title, price])

    if data:
        with open('books_data.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Title', 'Price'])
            writer.writerows(data)

        print("Data saved to books_data.csv")
    else:
        print("No data extracted. Verify the class names or inspect the HTML structure.")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
