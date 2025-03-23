import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


root_directory = "data2"
#add blue later
colors = ['blue', 'green', 'red', 'grey', 'black']

plt.figure(figsize=(12, 8))
legend_handles = []

for i, experiment_folder in enumerate(os.listdir(root_directory)):
    experiment_path = os.path.join(root_directory, experiment_folder)
    
    if os.path.isdir(experiment_path):
        dfs = []  
        for csv_file in os.listdir(experiment_path):
            csv_file_path = os.path.join(experiment_path, csv_file)
            if csv_file.endswith('.csv') and os.path.exists(csv_file_path):
                dfs.append(pd.read_csv(csv_file_path))
        
        combined_df = pd.concat(dfs)
        

        grouped = combined_df.groupby("num_time_steps")["average_return"]
        mean_return = grouped.mean()
        std_return = grouped.std()
        

        plt.plot(mean_return.index, mean_return, color=colors[i], label=experiment_folder, linewidth=2)
        plt.fill_between(mean_return.index, mean_return - std_return, mean_return + std_return, color=colors[i], alpha=0.3)
        legend_handles.append(Line2D([0], [0], color=colors[i], lw=4, label=experiment_folder))


plt.xlabel("Time Steps")
plt.ylabel("Average return")
plt.title("Policy Evaluation Across Experiments with Error Bands")


plt.xlim(left=0, right=500000) 

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=22)
plt.tight_layout()

# Save the plot
plt.savefig("experiments_with_error_bands.jpg")
plt.show()