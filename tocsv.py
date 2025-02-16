import csv
import os
import html
import re

def write_to_csv(file_name, file_text):
    with open('123.csv', 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        escaped_text = html.escape(file_text)
        csv_writer.writerow([f'{file_name}.jpg', escaped_text])

folder_path = r'D:\NIK_CINICHKA\cb\NN\Doc\output2'

for filename in os.listdir(folder_path):

    file_path = os.path.join(folder_path, filename)

    with open(file_path, 'r', encoding='utf-8') as txt_file:
        content = txt_file.read()
        content = re.sub(r'["\'\\;/\[\]{}()%^&#@|©`<’‘_№]', '', content)

    original_filename = filename.replace("processed_", "").split(".")[0]

    write_to_csv(file_name=original_filename, file_text=content)


