''' example to use Beautiful soup (bs4) to gather data '''


'''# import libraries:'''
from bs4 import BeautifulSoup
# Requests connects to a website which allows BeautifulSoup to interact and extract data
import requests

''' extracting data from the Trader Joe's website
    > Requests aquires the information of the page
    > then bs4 reads the page with an HTML parser'''
url = 'https://locations.traderjoes.com/'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

''' 
With the above information we want to inspect the webpage on our browser (CTRL + SHIFT + I) 
    this will show us the DOM of that element on the webpage and what we want is the href attribute
    which is the hyperlink to another part of the website
The below command collect all the href links within the above webpage into a list 
    so long as they are in the a class. 
unfortunately this include many unnecessary links '''
#gather all links
all_page_links = [link.get('href') for link in soup.find_all('a')]
# to filter these notice that all the relavant URLs are contained within <div id="contentbegin"> container
# gather all states used the html structure
results = soup.find(id="contentbegin")
states = [link.get('href') for link in results.find_all('a')]

''' gather all cities within a state 
Because the layout for each iterative page from states to cities is 
    the same we will wrap the above commands into a function '''

# import chain to flatten a list of lists into a single list
from itertools import chain

def get_content(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    # return the first instance of 'contentbegin' with the soup
    results = soup.find(id='contentbegin')
    # return all instances of a within the contentbegin block
    links = [link.get('href') for link in results.find_all('a')]
    return links

# create list of lists
# we can pass our URL list for each state directly into our function
cities = [get_content(state) for state in states]

# flatten the list of lists to create a single list
    # the * allows the funciton to work correctly, without it the function only yeilds the list of lists
    # the * tells the function that the list should be treated as multiple arguements and not one
flat_cities = list(chain(*cities))

''' gather list of locations in each city '''

# create list of lists
stores = [get_content(city) for city in flat_cities]
# flatten list to create one list and remove duplicates
    # the above lines of code returns two URLs for each city
# This occurs because there are two store links for each location,
        # one in the name the other in the view store details page
flat_stores = list(set(list(chain(*stores))))