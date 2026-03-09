import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

total_minutes = 24 * 60 
time_minutes = np.arange(total_minutes)
base_latency = 15

latency_noise = np.random.normal(loc=0, scale=1.2, size=total_minutes)
serviceb_latency = base_latency + latency_noise

start_minute = 600
duartion_minutes = 120
slope = 0.5

increase = slope * np.arange(duartion_minutes)
serviceb_latency[start_minute:start_minute + duartion_minutes] += increase

service_a_backlog = np.where(
    serviceb_latency > 40, 
    (serviceb_latency - 40)**2 + np.random.normal(0, 2, total_minutes), 
    np.random.randint(0, 5, size=total_minutes)
)

conditions = [
    (serviceb_latency < 25),
    (serviceb_latency >= 25) & (serviceb_latency < 40),
    (serviceb_latency >= 40) & (serviceb_latency < 70),
    (serviceb_latency >= 70)
]

choices = [0, 1, 2, 3]
status_level = np.select(conditions, choices, default=0)


data = {
    'minute': time_minutes,
    'latency_b': serviceb_latency,
    'backlog_a': service_a_backlog,
    'status': status_level
}

df = pd.DataFrame(data)
df.to_csv('service_health_data.csv', index=False)
print("Dataset generated successfully!")

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df["minute"], df["latency_b"], color='blue', label='Latency B (ms)')

ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Latency (ms)', color='blue')


ax2 = ax1.twinx()
ax2.plot(df['minute'], df['backlog_a'], color='red', alpha=0.6, label='Backlog A')
ax2.set_ylabel('Backlog Size (Units)', color='red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Anomaly Detection: Latency B vs Backlog A Correlation")
plt.grid(True, alpha=0.3)
plt.show()


from sklearn.model_selection import train_test_split

X = df[['latency_b', 'backlog_a']]
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))