class Student:
    def __init__(self, name, ID, email, embs, section, year):
        self.name = name
        self.ID = ID
        self.email = email
        self.embs = embs
        self.section = section
        self.year = year


class TA:
    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password


class Subject:
    def __init__(self, ID_name, year):
        self.ID_name = ID_name
        self.year = year
