This project demonstrates how to use Machine Learning to identify system failures before they crash the entire stack. Specifically, it monitors how high latency in one service (Service B) causes a request bottleneck (Backlog) in its dependent service (Service A).

In distributed systems, a single slow service can trigger a cascading failure. I generated a synthetic dataset that simulates 24 hours of traffic, including a critical incident at 10:00 AM where latency spikes and the backlog explodes.

Tech Stack

Python (Pandas, NumPy)
Scikit-Learn (Decision Tree Classifier)
Matplotlib (Dual-axis visualization)



Key Metrics & Visualization
The model analyzes the correlation between:
Latency B (ms): The root cause.
Backlog A: The symptom of the failure.

How the Model Works
I used a Decision Tree Classifier to categorize the system state into four levels: Healthy, Warning, Critical, and Down. By splitting the data into training and testing sets, the model learned to recognize the exact threshold where a "slow system" becomes a failed system.

