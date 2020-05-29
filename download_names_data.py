  url = "https://download.pytorch.org/tutorial/data.zip"
  r = requests.get(url, allow_redirects=True)
import requests 
import glob
import zipfile
import os

def download_extract_names_data():
  url = "https://download.pytorch.org/tutorial/data.zip"
  r = requests.get(url, allow_redirects=True)

  open('data.zip', 'wb').write(r.content)
  with zipfile.ZipFile("data.zip","r") as zip_ref:
      zip_ref.extractall("data")

download_extract_names_data()