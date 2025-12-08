# [Purpose] <> "The purpose of this file is act as a script for dictating the pipeline from data collection/prep to model dev/training"

import sys
import os
import pdb as pb
from ..MLLifecycle.DataCollectionAndPreparation import newVersionSet1FinalCopy as DP
from ..MLLifecycle.ModelDevelopmentAndTraining import set1FinalCopy as MD
# Body of setting up paths to access files external to script file
current_dir = os.path.dirname(__file__)
module_dir1 = os.path.join(current_dir,'DataCollectionAndPreparation/newVersionSet1FinalCopy.py')
module_dir2 = os.path.join(current_dir, 'ModelDevelopmentAndTraining/set1FinalCopy.py')
# End of Body of setting up paths to access files external to script file
sys.path.insert(0,module_dir1)
sys.path.insert(1,module_dir2)
DemoMode = True
module_dir3 = os.path.join('C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/plots' if DemoMode == False else "ProjectRepo/plots")
sys.path.insert(2,module_dir3)
sys.path = [sys.path[i].replace("\\","/") for i in range(len(sys.path))]
# print("--DEBUGGING CHECKPOINT: Ensuring that imports work--")
# pb.set_trace()

def main(listOfCompanies, dateToPullFrmStock = None):
   # NOTE: listOfCompanies comes from user's companies that they want to choose from.
   companies = listOfCompanies #<-- Populating companies in question here
   # Here, call the .py files to establish pipeline.  
   # Data Preparation
   print("---Data Preparation---")
   DP.dataPrepDeriv(companies,dateToPullFrmStock) #<-- Writes dataframe needed to be digested by model
   #os.system("python MLLifecycle/DataPreparation/testDir/newVersionSet1FinalCopy.py")
   # End of Data Preparation
   print("---End of Data Preparation---")
   # Model Development/Training
   print("---Model Development and Training---")
   MD.ModelTrainingAndDevelopment() #<-- Responsible for writing model to file, to be ready to run its predictions. 
   # End of Model Development/Training
   print("---End of Model Development/Training---")
   return
companies: list[str] = [
    "GOOG","AAPL","AMZN","MSFT",
    "META","NVDA","TSLA","BRK-B","JPM",
    "V","MA","HD","NFLX","DIS",
    "PEP","KO","XOM","CVX","ADBE","CSCO"
]

# main(companies)