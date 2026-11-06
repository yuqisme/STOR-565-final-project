import pandas as pd
for name in ["train", "validation", "test"]:  
    df = pd.read_csv(f"data/random_split/{name}.csv") 
    print(name, df.shape, "frauds:", int(df["Class"].sum()),
          f"({df['Class'].mean():.4%})")