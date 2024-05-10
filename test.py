from transformers import pipeline
import pandas as pd
 
sentiment_model = pipeline(model="jayllan23/ISY503-sentiment_analysis2")
data = sentiment_model(["I love this move", "This movie sucks!"])
df = pd.DataFrame(data)

# Get label
labels = df['label'].tolist()

# Get score
scores = df['score'].tolist()

print(f"Labels: {labels}")
print(f"Scores: {scores}")