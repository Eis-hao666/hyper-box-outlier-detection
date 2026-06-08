from scipy.io import loadmat

import pandas as pd

mat = loadmat('ionosphere.mat')
print(mat.keys())

# X = mat['trandata']

X = mat['X']
y = mat['y']

df = pd.DataFrame(X)
df['label'] = y.ravel()
# df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

print(df)
df.to_csv('ionosphere.txt', sep='\t', index=False)
