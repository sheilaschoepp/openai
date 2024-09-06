import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


root_directory = "data"

colors = ['blue', 'green', 'red', 'grey', 'black']

plt.figure(figsize=(16, 6))

legend_handles = []

for i, experiment_folder in enumerate(os.listdir(root_directory)):
    experiment_path = os.path.join(root_directory, experiment_folder)
    
    if os.path.isdir(experiment_path):
        for csv_file in os.listdir(experiment_path):
            csv_file_path = os.path.join(experiment_path, csv_file)
            
            if csv_file.endswith('.csv') and os.path.exists(csv_file_path):
                df = pd.read_csv(csv_file_path)
                
                plt.plot(df["num_time_steps"], df["average_return"], color=colors[i], alpha=0.6, linewidth=1.5)  # Increased line thickness
        
        legend_handles.append(Line2D([0], [0], color=colors[i], lw=4, label=experiment_folder))  # Increased legend line thickness

plt.xlabel("Time Steps")
plt.ylabel("Average Return")
plt.title("Policy Evaluation Across 75 Experiments")

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
plt.xlim(left=0, right=150000)


plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

plt.tight_layout()

plt.savefig("combined_experiments_evaluation_thicker_lines.jpg")
plt.show()