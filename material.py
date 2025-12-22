import random

class Material:
    def __init__(self, e1, e2, e3):
        self.elements = [e1, e2, e3]

    # よく使う合計・参照
    @property
    def m_total(self):
        return sum(e.m for e in self.elements)

    @property
    def b_total(self):
        return sum(e.b for e in self.elements)

    @property
    def r_total(self):
        return sum(e.r for e in self.elements)

    @property
    def k_total(self):
        return sum(e.k for e in self.elements)

    # 性能計算メソッド
    def calc_strength(self):
        e1, e2, e3 = self.elements
        return 2*e1.m + 1.5*e2.m + 0.5*e3.m + (e1.b * e2.b)

    def calc_heat(self):
        e1, e2, e3 = self.elements
        return (e1.t**2 + e2.t**2 + e3.t**2) - 0.1*(self.m_total)

    def calc_stability(self):
        return self.b_total - 0.8*(self.r_total)

    def calc_workability(self):
        return 100 / (1 + self.m_total) - 0.2*(self.b_total)

    def calc_corrosion(self):
        e1, e2, e3 = self.elements
        return 50 - (e1.e**2 + e2.e**2 + e3.e**2)

    def calc_quantum_drift(self):
        return self.k_total * random.uniform(-1, 1)

    def summary(self):
        return {
            "strength": self.calc_strength(),
            "heat": self.calc_heat(),
            "stability": self.calc_stability(),
            "workability": self.calc_workability(),
            "corrosion": self.calc_corrosion(),
            "quantum_drift": self.calc_quantum_drift()
        }
