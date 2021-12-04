import os, re

file_path = r'C:\Users\admin\Desktop\crawl\texts'
file_path2 = r'C:\Users\admin\Desktop\new_data'
file_list = os.listdir(file_path)
for id in file_list:
    file_name = f'{file_path}/{id}'
    with open(file_name, 'r', encoding='gbk') as f:
        bigString = f.read()
        new_string = re.sub('<[^<]+?>', '', bigString).strip()
    file_name = f'{file_path2}/{id}'
    with open(file_name, 'w', encoding='gbk') as f2:
        # token_list = re.split('\n', new_string)
        f2.write(new_string)
