import json
import os
from dotenv import load_dotenv

from examples.sorting.sorting_032 import SortingParser, SortingPrompter, utils
from graph_of_thoughts import controller, operations

load_dotenv()

config_json = 'config.json'
# Add API key to the JSON file
with open(config_json, 'r') as file:
    data = json.load(file)

data['chatgpt']['api_key'] == os.getenv("OPENAI_API_KEY")

# Write the modified data back to the JSON file
with open(config_json, 'w') as file:
    json.dump(data, file, indent=4)

# LC start
# Create the Graph of Operations
gop = operations.GraphOfOperations()
gop.append_operation(operations.Generate())
gop.append_operation(operations.Score(scoring_function=utils.num_errors))
gop.append_operation(operations.GroundTruth(utils.test_sorting))

# Configure the Language Model (Assumes config.json is in the current directory with OpenAI API key)
lm = controller.ChatGPT("config.json", model_name="chatgpt")

# Create the Controller
ctrl = controller.Controller(
  lm, 
  gop, 
  SortingPrompter(), 
  SortingParser(),
  # The following dictionary is used to configure the initial thought state
  {
    "original": to_be_sorted,
    "current": "",
    "method": "cot"
  }
)

# Run the Controller and generate the output graph
ctrl.run()
ctrl.output_graph("output_cot.json")