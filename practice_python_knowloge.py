import sys

a = [1, 2, 3]
b = a
print(f"{id(a)}")
print(f"{id(b)}")
print(f"{sys.getrefcount(a) - 1}")
b.append(4)
print(a)

s = "hello"
print(id(s))
s = s + "world"
print(id(s))

import gc

print(len(gc.get_objects()))


x = "global"

def outer():
    x = "enclosing"
    def inner():
        x = "local"
        print(x)

    inner()
    print()

outer()
print(x)

a = 10
b = a
a = a + 10

def iteral():
    a = 10
    b = a + 1
    c = a * b
    if c <= 3 :
        print(f"c = {c}")

    else:
        for i in range(10):
            print([i])

    try:
        print(f"{id(a) == id(c)} ")

    except:
        print(f"this way is error")

    def caculater(iteral):
        a = 9
        b = c * a
        if a == b:
            while a <= 3:
                a += 1
                print(f"{a}")
            else:
                return a

    return a

a = iteral()

import numpy as np
import pandas as pd

class HertDisease:
    """Hert disease prediction system"""
    def __init__(self):
        self.model =None
        self.scaler = StandardScaler()
        self.feature_name = [
            'age', 'sex', 'cp'
        ]
        self.feature_description = {
            'cp' : '0-3'
        }
    def create_sample_dataset(self, n_samples=300):
        np.random.seed(42)

        data = {
            'age':np.random.randint(29, 80, n_samples),
            'cp': np.random.randint(0, 2, n_samples)
        }

        df = pd.DataFrame(data)


class toolchaion:
    """tool chaion"""
    def __init__(self):
        self.dependencies = {
            'search_pubmed': ToolDependency(
                name="search_pubmed",
                depends_on=None,
                can_parallel=True,
                priority=Toolpriority.HIGH
            ),
            "fetch_pubmed": ToolDependency(
                name="fetch_full_text_links",
                depends_on=["search_pubmed"],
                can_parallel=True,
                priority=toolpriority.Low,
                timeout=20
            )
        }

    def can_execute_parallel(self, tool_cal:list[dict]) -> bool:
        if len(tool_cal) <= 1:
            return False
        for call in tool_cal:
            tool_name = call['function']['name']
            if tool_name in self.dependencies:
                if not self.dependencies[tool_name].can_parallel:
                    return False

        return True
    
