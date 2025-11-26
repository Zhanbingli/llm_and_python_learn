# import os

# cwd = os.getcwd() #resuent direction

# data_path = os.path.join(cwd, "my",'items.csv')

# os.makedirs(os.path.dirname(data_path), exist_ok=True)

# entries = os.listdir(os.path.dirname(data_path))

# for name in entries:
#     full = os.path.join(os.path.dirname(data_path), name)
#     if os.path.isdir(full):
#         print('dir',name)
#     elif os.path.isfile(full):
#         print('file',name)

# if os.path.exists(data_path):
#     info = os.stat(data_path)
#     print('size bytes:', info.st_size)
#     print('permissiions (oct):', oct(info.st_mode & 0o777))

# os.rename('old.txt', 'archived/old.txt')

# if os.path.exists('tmp.log'):
#     os.remove('tmp.log')

# for root, dirs, files in os.walk(cwd):
#     for fname in files:
#         print(os.path.join(root, fname))

# api_key = os.environ.get('API_KEY', '')
# os.environ['MOOE'] = "dev"

# import sys

# if not api_key:
#     sys.exit('missing API_KEY')

from pathlib import Path
p = Path("my")
p.parent.mkdir(parents=True, exist_ok=True)

if p.exists():
    size = p.stat().st_size

import json, csv
data = json.load(open('cfg.json', encoding='utf-8'))

row = list(csv.DictReader(open('data.csv', encoding='utf-8')))

from datetime import datetime, timezone, timedelta
now = datetime.now(timezone.utc)
print(now)

import re

m = re.search(r'\d+','a123b'); num = int(m.group())
