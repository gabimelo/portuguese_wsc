# flake8: noqa

# Path to each file whose English names will be replaced
paths_to_files = ['portuguese_wsc.json', 'portuguese_wsc.html']

# Dict with a Portuguese name for each English name
dict_names = {
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

for path in paths_to_files:
    # Read in the file
    with open(path, 'r') as file:
        filedata = file.read()
    
    # Replace all names
    print("At", path, ":")
    for english_name in reversed(sorted(dict_names.keys())):  # It will not replace "Ann" in "Anne" or "Anna", for instance
        if dict_names[english_name] not in filedata:
            print(english_name, "was replaced by", dict_names[english_name])
            filedata = filedata.replace(english_name, dict_names[english_name])

    print()
    # Write the file out again
    with open(path, 'w') as file:
        file.write(filedata)
