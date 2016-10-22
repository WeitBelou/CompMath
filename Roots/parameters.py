class Parameters:
    def __init__(self):
        self._parameters = dict()

    def parse_file(self, in_file_name: str):
        in_file = open(in_file_name)
        self._parameters = {s.strip().split('=')[0]: float(s.strip().split('=')[1])
                            for s in in_file.readlines() if s.strip()}

    def __getitem__(self, item):
        return self._parameters[item]