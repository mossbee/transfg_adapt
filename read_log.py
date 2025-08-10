from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# Load the log file
ea = EventAccumulator('logs/aug')
ea.Reload()

# Get all scalar events
scalars = ea.Scalars('train/loss')  # Replace with metric name
data = [(s.step, s.value) for s in scalars]
df = pd.DataFrame(data, columns=['step', 'value'])
print(df)