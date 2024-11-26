from brisque import BRISQUE as CalcBRISQUE

class BRISQUE(CalcBRISQUE):
    def __init__(self):
        super(BRISQUE, self).__init__()

    def calculate(self, img1, **kwargs):
        return self.get_score(img1)