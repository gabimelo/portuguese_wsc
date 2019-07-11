# flake8: noqa

from src.consts import WINOGRAD_SCHEMAS_FILE, WINOGRAD_PT_HTML_SCHEMAS_FILE

# Path to each file whose English names will be replaced
new_paths_to_files = []

# Create files from the ones without name translations
paths_to_files = [WINOGRAD_PT_HTML_SCHEMAS_FILE, WINOGRAD_SCHEMAS_FILE]

for path in paths_to_files:
    # Read in the file
    with open(path, 'r') as file:
        filedata = file.read()

    new_path = path.replace('.html', '_portuguese_names.html').replace('.json', '_portuguese_names.json')
    new_paths_to_files.append(new_path)

    with open(new_path, 'w') as file:
        file.write(filedata)

# Dict with a Portuguese name for each English name
dict_names = {
    # feminine names
    "Adams":   "Larissa",
    "Alice":   "Aline",
    "Amy":     "Camila",
    "Andrea":  "Francisca",
    "Ann":     "Adriana",
    "Anna":    "Ana",
    "Anne":    "Amanda",
    "Beth":    "Bruna",
    "Donna":   "Daniela",
    "Emma":    "Fernanda",
    "Grace":   "Eduarda",
    "Jade":    "Flavia",
    "Jane":    "Gabriela",
    "Janie":   "Juliana",
    "Joan":    "Jéssica",
    "Kate":    "Sofia",
    "Lily":    "Beatriz",
    "Lucy":    "Luciana",
    "Mary":    "Maria",
    "Pam":     "Patrícia",
    "Rebecca": "Rebeca",
    "Sally":   "Marcia",
    "Sue":     "Sandra",
    "Susan":   "Vanessa",
    # masculine names
    "Adam":    "Antonio",
    "Bill":    "André",
    "Billy":   "Arthur",
    "Bob":     "Bruno",
    "Carl":    "Carlos",
    "Charlie": "Fabio",
    "Cooper":  "Leonardo",
    "Dan ":    "Daniel ",
    "Eric":    "Manoel",
    "Ethan":   "Eduardo",
    "Frank":   "Felipe",
    "Fred":    "Francisco",
    "George ": "Gabriel ",
    "George,": "Gabriel,",
    "Joe":     "Jeorge",
    "James":   "Gustavo",
    "Jim":     "Guilherme",
    "Joe":     "Jorge",
    "Joey":    "José",
    "John":    "João",
    "Kevin":   "Luiz",
    "Luke":    "Lucas",
    "Mark":    "Marcos",
    "Martin":  "Marcelo",
    "Ollie":   "Mateus",
    "Paul":    "Paulo",
    "Pete":    "Pedro",
    "Ralph":   "Rafael",
    "Ray":     "Raimundo",
    "Robert":  "Roberto",
    "Sam":     "Samuel",
    "Sid":     "Ricardo",
    "Steve":   "Rodrigo",
    "Timmy":   "Tiago",
    "Thomson": "Fernando",
    "Toby":    "Sebastião",
    "Tom":     "Vinícius",
    "Tommy":   "Vitor"
}

for path in new_paths_to_files:
    # Read in the file
    with open(path, 'r') as file:
        filedata = file.read()
    
    # Replace all names
    print("At", path, ":")
    count = 0
    for english_name in reversed(sorted(dict_names.keys())):  # It will not replace "Ann" in "Anne" or "Anna", for instance
#        if dict_names[english_name] not in filedata:
        print(english_name, "was replaced by", dict_names[english_name])
        filedata = filedata.replace(english_name, dict_names[english_name])
        count += 1

    print('{} names were replaced. Expected {} substitutions.'.format(count, len(dict_names)))
    # Write the file out again
    with open(path, 'w') as file:
        file.write(filedata)
