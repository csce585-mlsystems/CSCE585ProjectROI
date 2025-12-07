# Purpose: This file will be responsible for producing all the plots that come about in the experiment. 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import pdb as pb

# Body of sequentially outputing the plots in question
DemoMode = True

module_dir3 = os.path.join('C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo' if DemoMode == False else 'ProjectReop')
sys.path.insert(0,module_dir3)
module_dir4 = os.path.join('C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/projectCode/MLLifecycle/ModelDevelopmentAndTraining' if DemoMode == False else 'projectCode/MLLifecycle/ModelDevelopmentAndTraining')
sys.path.insert(1,module_dir4)
sys.path = [x.replace("\\","/") for x in sys.path]
numOfExperiments = 4 #<-- update this if neccessary
numOfImages = 2*numOfExperiments #<-- update this later on
listOfFilePathsReffingPlots = [{"filePath": f"{sys.path[1]}/plots/Model{'Accuracy' if x % 2 == 0 else 'Loss'}Plot#1"} for x in range(numOfImages)]

# NOTE: For this to work, will have to add code in model file responsible for writing plot
# to a file!
print("---NOTE: Files referenced here, can be found in plots in ProjectRepo Directory folder")
print("---PRINTING....---")
isAccuracy = 0
for i in range(len(listOfFilePathsReffingPlots)):
    for j in ["Baseline", "Experiment#1", "Experiment#2", "Experiment#3"]:
        # print("---DEBUGGING CHECKPOINT: TESTING OUT RESULTS.py to ensure it works correctly---")
        # pb.set_trace()
        # img = mpimg.imread(image_path)
        print(f"---Model{('Accuracy' if isAccuracy %2 == 0 else 'Loss')}_{j}---")
        img = mpimg.imread(f"{listOfFilePathsReffingPlots[i]["filePath"]}_{j}.png")
        plt.title(f"Model{('Accuracy' if isAccuracy %2 == 0 else 'Loss')}_{j}")
        plt.figure(i)
        plt.imshow(img)
        isAccuracy += 1 
        # Line of code that writes image to a file in directory to be seen by viewers
        #plt.close()
        print(f"---End of Model{('Accuracy' if isAccuracy %2 == 0 else 'Loss')}_{j}---")

print("---PRINTING COMPLETE!---")
plt.show()

# End of Body of sequentially outputing the plots in question
