https://github.com/insightbuilder/python_de_learners_data/blob/main/code_script_notebooks/projects/exploring_dspy/playlist_assy.ipynb

import dspy
from dspy.teleprompt import BootstrapFewShot

# 1. Signature
class RouteSignature(dspy.Signature):
    query: str
    route: str  # Output: 'structured' or 'unstructured'

# 2. Routing Module
class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RouteSignature)

    def forward(self, query):
        return self.predict(query=query)

# 3. Training Examples
trainset = [
    dspy.Example(query="What is the target price of Apple stock?", route="structured").with_inputs("query"),
    dspy.Example(query="Summarize Apple's recent earnings report.", route="unstructured").with_inputs("query"),
    dspy.Example(query="How much does the iPhone 15 cost?", route="structured").with_inputs("query"),
    dspy.Example(query="Give me a background on Tesla's business model.", route="unstructured").with_inputs("query"),
    dspy.Example(query="Expected cost of the new model?", route="structured").with_inputs("query"),
    dspy.Example(query="Explain the features of the new electric car.", route="unstructured").with_inputs("query"),
]

# 4. Configure DSPy LM
lm = dspy.OpenAI(model="gpt-3.5-turbo")  # Or use dspy.HFModel
dspy.settings.configure(lm=lm)

# 5. Compile using Teleprompter
teleprompter = BootstrapFewShot(metric="exact_match")
optimized_router = teleprompter.compile(RouterModule(), trainset)

# 6. Inference
queries = [
    "What is the target price of Reliance stock?",
    "Explain Tesla's market strategy.",
    "How much does a MacBook Air cost?",
    "Tell me about Amazon’s cloud business."
]

for q in queries:
    result = optimized_router(q)
    print(f"🔍 Query: {q}\n➡️ Route: {result.route}\n")




class SQLPoTSignature(dspy.Signature):
    query: str
    schema: str = dspy.InputField(desc="Schema of the SQL table")
    reasoning: str
    sql_query: str

class SQLPoTGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SQLPoTSignature)

    def forward(self, query):
        schema = "Table: my_table(id INT, text TEXT, url TEXT)"
        return self.predict(query=query, schema=schema)

output = sql_pot_gen("Find all employee details")
print("🧠 Reasoning:\n", output.reasoning)
print("📝 SQL Query:\n", output.sql_query)

🧠 Reasoning:
The query asks to find employee details. The table 'my_table' has text column which likely stores content.
We can search for the keyword 'employee' in the text column using LIKE.

📝 SQL Query:
SELECT text, url FROM my_table WHERE text LIKE '%employee%'


teleprompter = BootstrapFewShot(metric="execution_accuracy")  # or exact_match
optimized_sql_gen = teleprompter.compile(SQLPoTGenerator(), trainset)

https://colab.research.google.com/drive/1LtsJjPofGZXfYRMrRtIQEQ_AVh0i3Ncq?usp=sharing

