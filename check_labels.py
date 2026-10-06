import csv
with open('fake_recogna_limpo.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    found_0, found_1 = False, False
    for row in reader:
        if row['Classe'] == '0' and not found_0:
            print("CLASSE 0:", row['Titulo'])
            found_0 = True
        if row['Classe'] == '1' and not found_1:
            print("CLASSE 1:", row['Titulo'])
            found_1 = True
        if found_0 and found_1: break
