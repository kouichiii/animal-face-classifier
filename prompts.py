CLASSIFY_PROMPT = """
# Task
You are an expert in facial analysis. Based on your own general knowledge of "dog-like" and "cat-like" facial archetypes, determine which one the person in the image resembles more.

# Instructions
1.  You must make a definitive choice between "dog_like" or "cat_like".
2.  Your reasoning must clearly state which specific facial features (e.g., eye shape, jawline, nose bridge, overall impression) led you to your conclusion.

# Output Format
Your response MUST be in the following format, and nothing else.

Decision: [dog_like or cat_like]
Reason: [Your detailed reasoning here, ensuring to mention the specific features that were the deciding factors.]
"""


