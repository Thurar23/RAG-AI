import json
import os

with open('acf.json', 'r', encoding='utf-8-sig') as f:
    bible = json.load(f)

os.makedirs('versiculos', exist_ok=True)

for ert in bible:
    book_name = ert["name"]
    book_abbr = ert["abbrev"]
    chapters = ert["chapters"]

    for index_cap, verse_cap in enumerate(chapters):
        chapter_num = index_cap + 1

        for index_verse, text_verse in enumerate(verse_cap):
            verse_num = index_verse + 1

            document = {
                "livro": book_name,
                "abreviacao": book_abbr,
                "capitulo": chapter_num,
                "versiculo": verse_num,
                "texto": text_verse 
            }

            name = f"{book_abbr}_{chapter_num}_{verse_num}.json"
            file = os.path.join('versiculos', name)

            with open(file, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
print("finished")

archi = os.listdir('versiculos')
print(len(archi))