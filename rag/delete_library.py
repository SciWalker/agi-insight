import os
from llmware.library import Library
from llmware.retrieval import Query
from llmware.prompts import Prompt
from llmware.setup import Setup
from llmware.models import ModelCatalog
from llmware.configs import LLMWareConfig
import time
LLMWareConfig().set_active_db("sqlite")
# library = Library().create_new_library("Test1234")
# sample_files_path = Setup().load_sample_files()

# library.add_files(os.path.join("data","check-addon-verif"))
print(Library().get_all_library_cards())
library = Library().load_library("Test123")
library.delete_library(confirm_delete=True)
print(Library().get_all_library_cards())