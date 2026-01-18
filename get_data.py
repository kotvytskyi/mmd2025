import urllib.request
import zipfile
import os
import shutil

url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
zip_path = "ml-1m.zip"
extract_dir = "data/"

if os.path.exists(extract_dir + "movies.dat"):
    print(f"MovieLens 1M dataset already exists in '{extract_dir}'. Skipping download.")
    exit(0)

os.makedirs(extract_dir, exist_ok=True)

urllib.request.urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("temp/")

for item in os.listdir("temp/ml-1m"):
    shutil.move(f"temp/ml-1m/{item}", extract_dir)

shutil.rmtree("temp/")
os.remove(zip_path)

shutil.copytree("data/", "part1/data/", dirs_exist_ok=True)
shutil.copytree("data/", "part2/data/", dirs_exist_ok=True)

print(f"Downloaded MovieLens 1M to '{extract_dir}' and copied the dataset to 'part1/data/' and 'part2/data/'.")
