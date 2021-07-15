''' example to use Beautiful soup (bs4) to gather data
    The scenario is we are buying a home and want it to suit our needs
    requirements:
    1. near trader joe's
    2. good internet speed in city
    3. good rating for piblic transportation systems'''


'''# import libraries:'''
from bs4 import BeautifulSoup
# Requests connects to a website which allows BeautifulSoup to interact and extract data
import requests
import pandas as pd

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
all_stores = list(set(list(chain(*stores))))

''' Gather information from each store
    1. get store name
    2. get address
    '''

def get_name(url):
    # request page
    page = requests.get(url)
    # parse content into html
    soup = BeautifulSoup(page.content, 'html.parser')
    # filter the subset of content we want
    results = soup.find(id='contentbegin')
    # within subset find and report the text from the h1title class
    store_name = results.find('div', class_='h1title').get_text()
        # why is there an _ after class? function doesn't work without it? what does it do?
    return store_name

def get_address(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find(id='contentbegin')
    # the address is in an unusable format and requires:
        # removal of indents (\t)
        # stripping of spaces with the strip() function
        # splitting of content into a list split at new paragraph symbol (\n)
    address = results.find('div', class_='addressline').get_text().replace('\t',"").strip().split('\n')
    # further strip white space for each element of the list we created
    address = [a.strip() for a in address]
    # remove the empty string elements with the filter command
    address = list(filter(None, address))
    return address

# the above function work independently
# therefore now we want to combine them to have only 1 request for each piece of info


def get_info(url):
    """
    Returns the information of interest for a given Trader Joe's store
    Args:
        url (list): URL of the store
    Returns:
        store_info  (list): Contains the store's name, city, state, zip, landline,
                            cell phone, and URL
    """
    # request page
    page = requests.get(url)
    # parse content into html
    soup = BeautifulSoup(page.content, 'html.parser')
    # filter the subset of content we want
    results = soup.find(id='contentbegin')
    # within subset find and report the text from the h1title class
    store_name = results.find('div', class_='h1title').get_text()
    # Get address and reformat
    address = results.find('div', class_='addressline').get_text().replace('\t','').strip().split('\n')
    address = [a.strip() for a in address]
    address = list(filter(None, address))
    store_info = [store_name] + address + [url]
    return store_info

# aquire information on all stores
all_store_info = [get_info(store) for store in all_stores]
info = ['store', 'street', 'city', 'state', 'zip', 'landline', 'mobile', 'website']
df = pd.DataFrame(all_store_info, columns=info)

# check df
df.count()
# which store have missing values?
df[df.website.isna()]
# The df has three missing values for website and mobile phone numbers
# the df has the website in the landline column
# copy this to the correct column
df.loc[df.website.isnull(), 'website'] = df['landline']

#set landline value to None
# find all rows with a null value, select the landline column, asign the values to None
df.loc[df.mobile.isnull(), 'landline'] = None