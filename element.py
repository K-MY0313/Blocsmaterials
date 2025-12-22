import random

class Element:
    def __init__(self, name):
        self.name = name
        
        # --- 名前ごとの個性定義 ---
        if name == "A":
            self.m = random.gauss(2, 0.5)      # 軽量
            self.e = random.uniform(-0.5, 0.5)
            self.b = random.uniform(2, 4)      # 結合しやすい
            self.t = random.uniform(1, 3)
            self.s = random.uniform(0.2, 0.8)
            self.r = random.uniform(1, 3)      # 反応性高め
            self.k = random.uniform(0.0, 0.3)  # 気まぐれ低め

        elif name == "B":
            self.m = random.gauss(8, 1.0)      # 重量級
            self.e = random.uniform(-1, 1)
            self.b = random.uniform(3, 5)      # 高結合
            self.t = random.uniform(0, 2)
            self.s = random.uniform(0.5, 1.0)
            self.r = random.uniform(0, 1)      # 反応性低め
            self.k = random.uniform(0.0, 0.2)

        elif name == "C":
            self.m = random.gauss(3, 0.5)
            self.e = random.gauss(0.0, 0.2)    # 電気性弱い
            self.b = random.uniform(1, 3)
            self.t = random.uniform(2, 4)
            self.s = random.uniform(0.0, 0.5)
            self.r = random.uniform(0, 1.5)
            self.k = random.uniform(0.0, 0.4)

        elif name == "D":
            self.m = random.gauss(5, 1.0)
            # 電気性が強く偏りやすい
            if random.random() < 0.5:
                self.e = random.uniform(0.5, 1.0)
            else:
                self.e = random.uniform(-1.0, -0.5)
            self.b = random.uniform(0, 2)
            self.t = random.uniform(1, 4)
            self.s = random.uniform(0.2, 0.7)
            self.r = random.uniform(1, 3)
            self.k = random.uniform(0.2, 0.6)

        else:  # その他の元素は“汎用型ランダム”
            self.m = random.uniform(0, 10)
            self.e = random.uniform(-1, 1)
            self.b = random.uniform(0, 5)
            self.t = random.uniform(0, 4)
            self.s = random.uniform(0, 1)
            self.r = random.uniform(0, 3)
            self.k = random.uniform(0, 1)

    def to_dict(self):
        return {
            "name": self.name,
            "m": self.m,
            "e": self.e,
            "b": self.b,
            "t": self.t,
            "s": self.s,
            "r": self.r,
            "k": self.k
        }
