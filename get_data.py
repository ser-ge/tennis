import pandas as pd 
from bs4 import BeautifulSoup
import wget
from zipfile import ZipFile
import os

# os.mkdir('zip_files')
# os.mkdir('tennis_data')

def download_zips(url, dr_zips, dr_unzips):
	zip_file = os.path.join(dr_zips, os.path.basename(url))
	print(zip_file)
	wget.download(url, zip_file)
	with ZipFile(zip_file, 'r') as myzip:
		myzip.extractall(dr_unzips)

def merge

start = 2000
stop = 2019

for year in range(start,stop+1):
	url = 'http://www.tennis-data.co.uk/{year}/{year}.zip'.format(year=str(year))
	download_zips(url, 'zip_files', 'tennis_data')



#with ZipFile('2019_data.zip', 'r') as myzip:
#	myzip.extractall()








