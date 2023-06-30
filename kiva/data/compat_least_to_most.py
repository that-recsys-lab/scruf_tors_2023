import pandas as pd

def main():
    d = pd.read_csv('compat.csv', header=None)
    dm = d.groupby([0], as_index=False).sum()

    final = pd.merge(d, dm, on=[0])
    final.columns = ['0', '1', '2', '3']
    final = final.sort_values(by=['3', '0'])
    final = final.drop(['3'], axis=1)

    final.to_csv("compat_least_to_most.csv", header=False, index=False)

if __name__ == "__main__":
    main()