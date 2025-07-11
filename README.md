https://github.com/insightbuilder/python_de_learners_data/blob/main/code_script_notebooks/projects/exploring_dspy/playlist_assy.ipynb
Za_D6Z5OYrjDxsEgWnDVQN-veob44klRgf0ln3fhT0kwN7pDSbXmOKHC-11PR6Lcox3kKXf0-7T3BlbkFJI8o2N7jKyGmPihKm98d-bEep1Yy6EYE420m8JJlrDOWXe9mTnDerBKNjxH76FevrEeYeW1-_EA

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


import fitz  # PyMuPDF
import os
from PIL import Image
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your API key

def pdf_to_images(pdf_path, image_folder="pdf_images"):
    os.makedirs(image_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        image_path = os.path.join(image_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    doc.close()
    return image_paths

def summarize_image(image_path):
    with open(image_path, "rb") as img_file:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Summarize the content of this PDF page."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_file.read().encode('base64').decode('utf-8')}}
                ]}
            ],
            temperature=0.5,
        )
    return response["choices"][0]["message"]["content"]

def delete_images(image_paths):
    for path in image_paths:
        os.remove(path)
    os.rmdir(os.path.dirname(image_paths[0]))

def summarize_pdf(pdf_path):
    print(f"Processing PDF: {pdf_path}")
    image_paths = pdf_to_images(pdf_path)
    
    full_summary = []
    for image_path in image_paths:
        print(f"Summarizing {image_path} ...")
        summary = summarize_image(image_path)
        full_summary.append(f"\n--- Page {image_paths.index(image_path) + 1} ---\n{summary}")

    delete_images(image_paths)
    return "\n".join(full_summary)

# Example usage
if _name_ == "_main_":
    pdf_file = "example.pdf"  # 🔁 Replace with your PDF path
    final_summary = summarize_pdf(pdf_file)
    print("\n📘 Final Summary of PDF:\n", final_summary)

