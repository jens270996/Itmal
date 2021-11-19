import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("spotify_data.csv")

# print(data.dtypes)
# print(data.columns)
print(data.where(data["explicit"]==1)["explicit"].count())
classical_data = data.where(data["genre"] == 0).dropna().drop(columns=["genre","mode","explicit","key","time_signature"]).reset_index(drop=True) #blue
jazz_data = data.where(data["genre"] == 1).dropna().drop(columns=["genre","mode","explicit","key","time_signature"]).reset_index(drop=True) #yellow
techno_data = data.where(data["genre"] == 2).dropna().drop(columns=["genre","mode","explicit","key","time_signature"]).reset_index(drop=True) #nyan
rock_data = data.where(data["genre"] == 3).dropna().drop(columns=["genre","mode","explicit","key","time_signature"]).reset_index(drop=True) #red

for outer_attribute in classical_data.columns:
    vals=[classical_data[outer_attribute].mean(),jazz_data[outer_attribute].mean(),techno_data[outer_attribute].mean(),rock_data[outer_attribute].mean()]
    plt.bar(height=vals,x=["classical","jazz","techno","rock"])
    title = outer_attribute+" Bar diagram"
    plt.title(title)
    ylabel=outer_attribute+" mean"
    plt.ylabel(ylabel=ylabel)
    plt.xlabel(xlabel="Genre")
    # ax.set_ylabel(classical_data[outer_attribute]+" mean")
    # ax.set_xlabel("Genre")
    # ax.set_title(classical_data[outer_attribute]+" bar diagram")
    
    plt.show()
    for inner_attribute in classical_data.columns:
        if inner_attribute == outer_attribute:
            break
        # plt.scatter(x=jazz_data[outer_attribute],y=jazz_data[inner_attribute],c="yellow")
        # plt.scatter(x=classical_data[outer_attribute],y=classical_data[inner_attribute],c="blue")
        # plt.scatter(x=rock_data[outer_attribute],y=rock_data[inner_attribute],c="red")
        # plt.scatter(x=techno_data[outer_attribute],y=techno_data[inner_attribute],c="fuchsia")
        # plt.xlabel(outer_attribute)
        # plt.ylabel(inner_attribute)
        # plt.title(outer_attribute+"/"+inner_attribute)
        #plt.show()


