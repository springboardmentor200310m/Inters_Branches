import pandas as pd

meta = pd.read_csv("TinySOL_metadata.csv")


meta["label"] = meta["Instrument (in full)"]     
meta["FileName"] = meta["Path"].apply(lambda x: x.split("/")[-1])

meta["png_name"] = meta["FileName"].str.replace(".wav", ".png")

label_dict = dict(zip(meta["png_name"], meta["label"]))

import pandas as pd
import os


meta = pd.read_csv("TinySOL_metadata.csv")

meta["FileName"] = meta["Path"].apply(lambda x: os.path.basename(x))

meta["PNGName"] = meta["FileName"].str.replace(".wav", ".png")

instrument_col = "Instrument (in full)"
intensity_col = "Dynamics"        # This is intensity

# Create final dataframe
df = meta[["PNGName", instrument_col, intensity_col]].rename(columns={
    "PNGName": "image",
    instrument_col: "instrument",
    intensity_col: "intensity"
})

df.to_csv("mel_labels.csv", index=False)

print("Saved mel_labels.csv")
print(df.head())
