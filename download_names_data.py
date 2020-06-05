#simple utility file to download the rnn names data onto the head node of the cluster
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

if __name__ = "__main__":
    download_extract_names_data()
